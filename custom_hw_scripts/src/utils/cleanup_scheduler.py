"""
==========================================================================
  CLEANUP SCHEDULER – Gerenciamento Automático de Armazenamento
==========================================================================

Módulo responsável pela limpeza automática de arquivos temporários
gerados pelo pipeline World-to-Mesh. Projetado para GPU alugada,
onde espaço em disco é limitado e caro.

Pulo do gato #1: Cada sessão gera ~200-500MB de arquivos temporários.
Em GPU alugada sem limpeza, o disco enche em poucas horas.

Estratégias de Limpeza:
  1. Imediata: Após download bem-sucedido pelo usuário
  2. TTL: Sessões não baixadas expiram após N horas (padrão: 1h)
  3. Periódica: Verificação a cada 15 minutos
  4. Emergencial: Quando disco > 90% cheio

Uso via terminal:
    python -m src.utils.cleanup_scheduler [--base-dir /tmp/world_to_mesh]
==========================================================================
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("world_to_mesh.cleanup")


# ── Constantes ─────────────────────────────────────────────────────────

DEFAULT_BASE_DIR = "/tmp/world_to_mesh"
DEFAULT_TTL_HOURS = 1.0
DEFAULT_CHECK_INTERVAL_SECONDS = 15 * 60  # 15 minutos
DEFAULT_DISK_THRESHOLD = 0.90  # 90% de uso do disco


class CleanupScheduler:
    """
    Gerenciador de limpeza automática para arquivos temporários.

    Ideal para ambientes de GPU alugada (RunPod, Vast.ai, Lambda)
    onde espaço em disco é limitado.

    Exemplo:
        >>> cleaner = CleanupScheduler("/tmp/world_to_mesh")
        >>> session_id = cleaner.create_session()
        >>> # ... pipeline gera arquivos na sessão ...
        >>> cleaner.schedule_cleanup(session_id, ttl_hours=1.0)
        >>> # Após download:
        >>> cleaner.immediate_cleanup(session_id)
    """

    def __init__(
        self,
        base_dir: str = DEFAULT_BASE_DIR,
        ttl_hours: float = DEFAULT_TTL_HOURS,
        check_interval: int = DEFAULT_CHECK_INTERVAL_SECONDS,
        disk_threshold: float = DEFAULT_DISK_THRESHOLD,
    ):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.ttl_hours = ttl_hours
        self.check_interval = check_interval
        self.disk_threshold = disk_threshold

        # Arquivo de tracking das sessões
        self._queue_file = self.base_dir / "cleanup_queue.json"

        # Thread de limpeza periódica
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Garantir diretórios base
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Carregar fila existente (para sobreviver a reinícios)
        self._queue = self._load_queue()

    # ── Gerenciamento de Sessões ───────────────────────────────────────

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Cria uma nova sessão com diretório próprio.

        Args:
            session_id: ID customizado (ou auto-gera baseado em timestamp).

        Returns:
            str: ID da sessão criada.
        """
        if session_id is None:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Salvar metadata da sessão
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"📁 Sessão criada: {session_id} em {session_dir}")
        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        """Retorna o diretório de uma sessão."""
        return self.sessions_dir / session_id

    # ── Agendamento de Limpeza ─────────────────────────────────────────

    def schedule_cleanup(
        self, session_id: str, ttl_hours: Optional[float] = None
    ):
        """
        Agenda limpeza futura de uma sessão.

        A sessão será removida automaticamente após o TTL expirar.

        Args:
            session_id: ID da sessão a limpar
            ttl_hours: Tempo até limpeza (None = usa padrão da instância)
        """
        ttl = ttl_hours or self.ttl_hours
        expire_at = time.time() + (ttl * 3600)

        self._queue[session_id] = {
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "expire_at": expire_at,
            "ttl_hours": ttl,
        }
        self._save_queue()

        logger.info(
            f"⏰ Limpeza agendada: sessão '{session_id}' "
            f"expira em {ttl:.1f}h"
        )

    def immediate_cleanup(self, session_id: str) -> bool:
        """
        Limpa uma sessão imediatamente (após download bem-sucedido).

        Args:
            session_id: ID da sessão a limpar.

        Returns:
            bool: True se limpeza foi bem-sucedida.
        """
        session_dir = self.sessions_dir / session_id

        if not session_dir.exists():
            logger.warning(f"Sessão '{session_id}' não encontrada para limpeza")
            return False

        try:
            # Calcular tamanho antes de deletar (para logging)
            size_mb = self._dir_size_mb(session_dir)

            shutil.rmtree(session_dir)

            # Remover da fila de agendamento
            self._queue.pop(session_id, None)
            self._save_queue()

            logger.info(
                f"🗑️  Sessão '{session_id}' limpa imediatamente "
                f"({size_mb:.1f} MB liberados)"
            )
            return True

        except Exception as e:
            logger.error(f"Erro ao limpar sessão '{session_id}': {e}")
            return False

    # ── Limpeza Periódica ──────────────────────────────────────────────

    def run_periodic_cleanup(self):
        """
        Executa uma rodada de limpeza periódica.

        Verifica:
        1. Sessões com TTL expirado
        2. Nível de uso do disco
        """
        now = time.time()
        expired_sessions = []

        # 1. Encontrar sessões expiradas
        for session_id, info in self._queue.items():
            if now >= info.get("expire_at", float("inf")):
                expired_sessions.append(session_id)

        # 2. Limpar sessões expiradas
        for session_id in expired_sessions:
            logger.info(f"⏰ TTL expirado para sessão '{session_id}'")
            self.immediate_cleanup(session_id)

        # 3. Verificar nível de disco
        disk_usage = self._get_disk_usage()
        if disk_usage and disk_usage > self.disk_threshold:
            logger.warning(
                f"⚠️  Disco em {disk_usage * 100:.1f}% "
                f"(threshold: {self.disk_threshold * 100:.0f}%)"
            )
            self.emergency_cleanup()

        # 4. Limpar sessões órfãs (sem entrada na fila)
        self._cleanup_orphan_sessions()

        if expired_sessions:
            logger.info(
                f"🧹 Limpeza periódica: {len(expired_sessions)} sessões removidas"
            )

    def emergency_cleanup(self):
        """
        Limpeza emergencial quando disco está muito cheio.

        Remove as sessões mais antigas primeiro (FIFO).
        """
        logger.warning("🚨 Iniciando limpeza emergencial...")

        # Listar todas as sessões por data de criação
        sessions = []
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                meta_path = session_dir / "metadata.json"
                created = 0
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        created = meta.get("created_at", "")
                    except Exception:
                        pass
                sessions.append((created, session_dir.name))

        # Ordenar por data (mais antigos primeiro)
        sessions.sort(key=lambda x: x[0])

        # Remover sessões até disco ficar abaixo do threshold
        removed = 0
        for _, session_id in sessions:
            self.immediate_cleanup(session_id)
            removed += 1

            disk_usage = self._get_disk_usage()
            if disk_usage and disk_usage < self.disk_threshold * 0.8:
                break  # Alvo: 80% do threshold para dar margem

        logger.warning(
            f"🚨 Limpeza emergencial concluída: "
            f"{removed} sessões removidas"
        )

    # ── Background Thread ──────────────────────────────────────────────

    def start_background_cleanup(self):
        """
        Inicia thread de limpeza periódica em background.

        A thread roda a cada `check_interval` segundos.
        Chamar `stop_background_cleanup()` para parar.
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("Background cleanup já está rodando")
            return

        self._stop_event.clear()

        def _cleanup_loop():
            logger.info(
                f"🔄 Background cleanup iniciado "
                f"(intervalo: {self.check_interval}s)"
            )
            while not self._stop_event.is_set():
                try:
                    self.run_periodic_cleanup()
                except Exception as e:
                    logger.error(f"Erro na limpeza periódica: {e}")
                self._stop_event.wait(self.check_interval)
            logger.info("🔄 Background cleanup parado")

        self._cleanup_thread = threading.Thread(
            target=_cleanup_loop, daemon=True, name="cleanup-scheduler"
        )
        self._cleanup_thread.start()

    def stop_background_cleanup(self):
        """Para a thread de limpeza periódica."""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    # ── Utilidades ─────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Retorna status atual do scheduler.

        Útil para monitoramento remoto via terminal/SSH.
        """
        sessions = list(self.sessions_dir.iterdir()) if self.sessions_dir.exists() else []
        active_sessions = [d.name for d in sessions if d.is_dir()]
        total_size = sum(
            self._dir_size_mb(d) for d in sessions if d.is_dir()
        )
        disk_usage = self._get_disk_usage()

        return {
            "base_dir": str(self.base_dir),
            "active_sessions": len(active_sessions),
            "scheduled_cleanups": len(self._queue),
            "total_size_mb": round(total_size, 2),
            "disk_usage_percent": round(disk_usage * 100, 1) if disk_usage else None,
            "ttl_hours": self.ttl_hours,
            "background_running": (
                self._cleanup_thread is not None
                and self._cleanup_thread.is_alive()
            ),
        }

    # ── Internos ───────────────────────────────────────────────────────

    def _load_queue(self) -> dict:
        """Carrega fila de limpeza do disco (sobrevive a reinícios)."""
        if self._queue_file.exists():
            try:
                with open(self._queue_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_queue(self):
        """Persiste fila de limpeza no disco."""
        try:
            with open(self._queue_file, "w", encoding="utf-8") as f:
                json.dump(self._queue, f, indent=2)
        except IOError as e:
            logger.error(f"Erro ao salvar fila de limpeza: {e}")

    def _dir_size_mb(self, directory: Path) -> float:
        """Calcula tamanho total de um diretório em MB."""
        total = 0
        try:
            for f in directory.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except (OSError, PermissionError):
            pass
        return total / (1024 * 1024)

    def _get_disk_usage(self) -> Optional[float]:
        """Retorna fração de uso do disco (0.0 a 1.0)."""
        try:
            usage = shutil.disk_usage(str(self.base_dir))
            return usage.used / usage.total
        except (OSError, FileNotFoundError):
            return None

    def _cleanup_orphan_sessions(self):
        """Remove sessões que existem no disco mas não na fila."""
        if not self.sessions_dir.exists():
            return

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name

            # Verificar se sessão tem metadata válido
            meta_path = session_dir / "metadata.json"
            if not meta_path.exists():
                # Sessão sem metadata = órfã, limpar
                logger.info(f"🧹 Removendo sessão órfã: {session_id}")
                self.immediate_cleanup(session_id)
                continue

            # Verificar se sessão é muito antiga (> 2x TTL)
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                created_str = meta.get("created_at", "")
                if created_str:
                    created = datetime.fromisoformat(created_str)
                    age_hours = (
                        datetime.now(timezone.utc) - created
                    ).total_seconds() / 3600
                    if age_hours > self.ttl_hours * 2:
                        logger.info(
                            f"🧹 Sessão expirada (2x TTL): {session_id} "
                            f"({age_hours:.1f}h)"
                        )
                        self.immediate_cleanup(session_id)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass


# ── CLI Quick-Test ─────────────────────────────────────────────────────
# Permite testar o scheduler via terminal:
#   python -m src.utils.cleanup_scheduler [--base-dir /tmp/world_to_mesh]

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cleanup Scheduler CLI")
    parser.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help="Diretório base para sessões temporárias",
    )
    parser.add_argument(
        "--ttl", type=float, default=DEFAULT_TTL_HOURS,
        help="TTL em horas para sessões",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Mostrar status atual",
    )
    parser.add_argument(
        "--cleanup-now", action="store_true",
        help="Executar limpeza imediata",
    )
    args = parser.parse_args()

    scheduler = CleanupScheduler(
        base_dir=args.base_dir, ttl_hours=args.ttl
    )

    if args.status:
        status = scheduler.get_status()
        print("\n📊 Status do Cleanup Scheduler:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    elif args.cleanup_now:
        print("🧹 Executando limpeza...")
        scheduler.run_periodic_cleanup()
        print("✅ Limpeza concluída")
    else:
        print("📊 Nenhuma ação especificada. Use --status ou --cleanup-now")
        status = scheduler.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
