"""
==========================================================================
  IMAGE QUALITY VALIDATOR – Validação de Qualidade de Imagem
==========================================================================

Módulo que valida a qualidade de imagens geradas pelo HunyuanWorld ANTES
de enviá-las ao Hunyuan3D-2 para reconstrução 3D. Isso economiza tempo
e custo de GPU ao evitar processamento de imagens ruins.

Pulo do gato #3: Cada validação custa ~1s CPU vs ~30s GPU no Hunyuan3D-2.
Em caso de falha, economiza 97% do tempo/custo de processamento.

Critérios de Validação:
  1. Dimensões mínimas (>= 256x256)
  2. Nitidez (Laplacian variance > threshold)
  3. Não é imagem completamente preta/branca
  4. Diversidade de cores (histogram distribution)
  5. Aspect ratio razoável

Uso via terminal:
    python -m src.utils.image_validator path/to/image.png
==========================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("world_to_mesh.image_validator")


# ── Constantes de Validação ────────────────────────────────────────────

DEFAULT_MIN_SIZE = 256          # Dimensão mínima em pixels
DEFAULT_SHARPNESS_THRESHOLD = 50.0  # Laplacian variance mínima
DEFAULT_MIN_COLOR_STD = 15.0    # Desvio padrão mínimo de cor (0-255)
DEFAULT_BLACK_WHITE_RATIO = 0.95  # Máx. % de pixels preto ou branco
DEFAULT_MAX_ASPECT_RATIO = 4.0  # Aspect ratio máximo (largura/altura)


class ImageQualityValidator:
    """
    Validador de qualidade de imagens geradas.

    Executa múltiplos checks rápidos (CPU-only) antes de enviar
    a imagem para processamento 3D caro (GPU).

    Exemplo:
        >>> validator = ImageQualityValidator()
        >>> is_ok, reason = validator.validate("output/frame_0001.png")
        >>> if not is_ok:
        ...     print(f"Imagem rejeitada: {reason}")
    """

    def __init__(
        self,
        min_size: int = DEFAULT_MIN_SIZE,
        sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
        min_color_std: float = DEFAULT_MIN_COLOR_STD,
        black_white_ratio: float = DEFAULT_BLACK_WHITE_RATIO,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
    ):
        self.min_size = min_size
        self.sharpness_threshold = sharpness_threshold
        self.min_color_std = min_color_std
        self.black_white_ratio = black_white_ratio
        self.max_aspect_ratio = max_aspect_ratio

    # ── API Principal ──────────────────────────────────────────────────

    def validate(self, image_path: str) -> tuple:
        """
        Valida a qualidade de uma imagem.

        Executa todos os checks em sequência. Para no primeiro erro
        para economizar tempo.

        Args:
            image_path: Caminho absoluto ou relativo da imagem.

        Returns:
            tuple: (is_valid: bool, report: dict)
                   report contém:
                     - "passed": bool
                     - "reason": str (motivo se falhou)
                     - "scores": dict com métricas individuais
                     - "checks": dict com resultado de cada check
        """
        path = Path(image_path)
        report = {
            "passed": False,
            "reason": "",
            "scores": {},
            "checks": {},
        }

        # ── Check 0: Arquivo existe? ──
        if not path.exists():
            report["reason"] = f"Arquivo não encontrado: {path}"
            logger.warning(report["reason"])
            return False, report

        # ── Carregar imagem ──
        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                report["reason"] = f"Falha ao decodificar imagem: {path}"
                logger.warning(report["reason"])
                return False, report
        except Exception as e:
            report["reason"] = f"Erro ao carregar imagem: {e}"
            logger.warning(report["reason"])
            return False, report

        h, w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Check 1: Dimensões mínimas ──
        size_ok = self._check_dimensions(w, h)
        report["checks"]["dimensions"] = size_ok
        report["scores"]["width"] = w
        report["scores"]["height"] = h
        if not size_ok:
            report["reason"] = (
                f"Dimensões insuficientes: {w}x{h} "
                f"(mínimo: {self.min_size}x{self.min_size})"
            )
            logger.warning(report["reason"])
            return False, report

        # ── Check 2: Aspect ratio ──
        ar_ok = self._check_aspect_ratio(w, h)
        report["checks"]["aspect_ratio"] = ar_ok
        report["scores"]["aspect_ratio"] = round(max(w, h) / max(min(w, h), 1), 2)
        if not ar_ok:
            report["reason"] = (
                f"Aspect ratio muito extremo: {report['scores']['aspect_ratio']} "
                f"(máximo: {self.max_aspect_ratio})"
            )
            logger.warning(report["reason"])
            return False, report

        # ── Check 3: Não é imagem preta / branca ──
        bw_ok, bw_ratio = self._check_not_blank(img_gray)
        report["checks"]["not_blank"] = bw_ok
        report["scores"]["blank_pixel_ratio"] = round(bw_ratio, 4)
        if not bw_ok:
            report["reason"] = (
                f"Imagem quase totalmente preta/branca "
                f"({bw_ratio * 100:.1f}% pixels extremos, "
                f"máximo: {self.black_white_ratio * 100:.0f}%)"
            )
            logger.warning(report["reason"])
            return False, report

        # ── Check 4: Nitidez suficiente ──
        sharpness = self._compute_sharpness(img_gray)
        sharp_ok = sharpness >= self.sharpness_threshold
        report["checks"]["sharpness"] = sharp_ok
        report["scores"]["sharpness"] = round(sharpness, 2)
        if not sharp_ok:
            report["reason"] = (
                f"Imagem muito borrada: sharpness={sharpness:.2f} "
                f"(mínimo: {self.sharpness_threshold})"
            )
            logger.warning(report["reason"])
            return False, report

        # ── Check 5: Diversidade de cores ──
        color_ok, color_std = self._check_color_diversity(img_rgb)
        report["checks"]["color_diversity"] = color_ok
        report["scores"]["color_std"] = round(color_std, 2)
        if not color_ok:
            report["reason"] = (
                f"Pouca diversidade de cores: std={color_std:.2f} "
                f"(mínimo: {self.min_color_std})"
            )
            logger.warning(report["reason"])
            return False, report

        # ── Todos os checks passaram ──
        report["passed"] = True
        report["reason"] = "Todos os critérios de qualidade atendidos"
        logger.info(
            f"✅ Imagem validada: {path.name} "
            f"(sharp={sharpness:.0f}, color_std={color_std:.1f}, "
            f"size={w}x{h})"
        )
        return True, report

    def get_quality_score(self, image_path: str) -> float:
        """
        Retorna um score numérico de qualidade (0-100).

        Útil para selecionar o melhor frame de uma sequência.
        Score mais alto = melhor qualidade.

        Args:
            image_path: Caminho da imagem.

        Returns:
            float: Score de 0 a 100.
        """
        path = Path(image_path)
        if not path.exists():
            return 0.0

        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                return 0.0
        except Exception:
            return 0.0

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Componentes do score (0-25 cada, total 0-100)
        sharpness = min(self._compute_sharpness(img_gray) / 500.0, 1.0) * 25.0

        _, color_std = self._check_color_diversity(img_rgb)
        color_score = min(color_std / 60.0, 1.0) * 25.0

        h, w = img_bgr.shape[:2]
        size_score = min((w * h) / (768 * 768), 1.0) * 25.0

        _, bw_ratio = self._check_not_blank(img_gray)
        content_score = (1.0 - bw_ratio) * 25.0

        total = sharpness + color_score + size_score + content_score
        return round(total, 2)

    # ── Checks Individuais ─────────────────────────────────────────────

    def _check_dimensions(self, width: int, height: int) -> bool:
        """Verifica se ambas as dimensões são >= min_size."""
        return width >= self.min_size and height >= self.min_size

    def _check_aspect_ratio(self, width: int, height: int) -> bool:
        """Verifica se aspect ratio não é muito extremo."""
        ratio = max(width, height) / max(min(width, height), 1)
        return ratio <= self.max_aspect_ratio

    def _check_not_blank(self, gray_image: np.ndarray) -> tuple:
        """
        Verifica se imagem NÃO é quase toda preta ou branca.

        Returns:
            tuple: (is_ok: bool, blank_ratio: float)
        """
        total_pixels = gray_image.size
        # Contar pixels muito escuros (< 10) ou muito claros (> 245)
        black_pixels = np.sum(gray_image < 10)
        white_pixels = np.sum(gray_image > 245)
        blank_ratio = (black_pixels + white_pixels) / total_pixels
        return blank_ratio < self.black_white_ratio, blank_ratio

    def _compute_sharpness(self, gray_image: np.ndarray) -> float:
        """
        Calcula nitidez usando variância do Laplaciano.

        Quanto maior o valor, mais nítida a imagem.
        Imagens borradas (blur) terão valores baixos.
        """
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())

    def _check_color_diversity(self, rgb_image: np.ndarray) -> tuple:
        """
        Verifica se a imagem tem variação de cores suficiente.

        Usa desvio padrão médio dos canais RGB como métrica.
        Imagens monocromáticas ou de uma cor só terão std baixo.

        Returns:
            tuple: (is_diverse: bool, mean_std: float)
        """
        # Desvio padrão por canal
        stds = [float(np.std(rgb_image[:, :, c])) for c in range(3)]
        mean_std = sum(stds) / 3.0
        return mean_std >= self.min_color_std, mean_std


# ── CLI Quick-Test ─────────────────────────────────────────────────────
# Permite testar o validador rapidamente via terminal:
#   python -m src.utils.image_validator path/to/image.png

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    validator = ImageQualityValidator()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n🔍 Validando: {image_path}")
        is_valid, report = validator.validate(image_path)
        score = validator.get_quality_score(image_path)

        print(f"\n{'✅ PASSOU' if is_valid else '❌ FALHOU'}")
        print(f"Motivo: {report['reason']}")
        print(f"Score geral: {score}/100")
        print(f"\nScores detalhados:")
        for key, value in report["scores"].items():
            check_status = report["checks"].get(key, "N/A")
            icon = "✅" if check_status is True else ("❌" if check_status is False else "➖")
            print(f"  {icon} {key}: {value}")
    else:
        print("Uso: python -m src.utils.image_validator <caminho_da_imagem>")
        print("Exemplo: python -m src.utils.image_validator output/frame_0001.png")
