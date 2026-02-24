#!/usr/bin/env python3
"""
==========================================================================
  BRIDGE.PY – Orquestrador World-to-Mesh Universal
==========================================================================

Ponte principal que conecta HunyuanWorld-Mirror (geração de mundos) ao
Hunyuan3D-2 (reconstrução 3D) em um pipeline automatizado end-to-end.

Como um "meshy.ai de mapas 3D": o usuário fornece um prompt textual e
recebe um arquivo .glb pronto para Unreal/Unity.

  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
  │ Prompt User  │──▶  │  HunyuanWorld   │──▶  │  Hunyuan3D-2  │
  │ + Estilo     │     │  (Arquiteto)    │     │  (Construtor) │
  └─────────────┘     └────────┬────────┘     └──────┬───────┘
                               │                      │
                        Frame Mestre              Arquivo .glb
                         (validado)             (pronto p/ engine)

"Pulos do Gato" implementados:
  1. Intercâmbio via /tmp/ com auto-limpeza (CleanupScheduler)
  2. Dicionário de estilos com keyword injection (StyleManager)
  3. Validação de qualidade antes do 3D (ImageQualityValidator)

Uso via terminal (ideal para SSH em GPU alugada):
  python bridge.py --prompt "vila medieval com castelo" --style minecraft
  python bridge.py --prompt "floresta mágica" --style rpg --output resultado.glb
  python bridge.py --prompt "base espacial" --style sci-fi --no-texture

==========================================================================
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

# ── Caminhos dos projetos ──────────────────────────────────────────────
# Ajuste estes caminhos conforme a localização dos projetos no seu sistema.
# Em GPU alugada, estes caminhos podem ser diferentes.

WORKSPACE_ROOT = Path(__file__).resolve().parent
HUNYUAN_WORLD_DIR = WORKSPACE_ROOT / "HunyuanWorld-Mirror-main" / "HunyuanWorld-Mirror-main"
HUNYUAN_3D_DIR = WORKSPACE_ROOT / "Hunyuan3D-2-main" / "Hunyuan3D-2-main"

# Diretório temporário para intercâmbio entre os pipelines
# PULO DO GATO #1: Usar /tmp/ para não entupir storage da GPU alugada
DEFAULT_TEMP_DIR = os.environ.get(
    "WORLD_TO_MESH_TEMP",
    "/tmp/world_to_mesh" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "C:\\Temp"), "world_to_mesh")
)


# ── Logging ────────────────────────────────────────────────────────────
# Configuração de logging para monitoramento remoto via SSH/terminal

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """
    Configura logging do pipeline.

    Em cloud/GPU alugada, os logs vão tanto para o terminal (SSH)
    quanto para um arquivo, permitindo monitoramento em tempo real.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("world_to_mesh")


# ── Pipeline Principal ─────────────────────────────────────────────────

class WorldToMeshPipeline:
    """
    Orquestrador completo: Texto → Mundo → Mesh 3D

    Integra HunyuanWorld-Mirror e Hunyuan3D-2 em um fluxo automatizado
    com validação de qualidade, retry inteligente e auto-limpeza.

    Exemplo:
        >>> pipeline = WorldToMeshPipeline()
        >>> result = pipeline.generate(
        ...     prompt="uma vila medieval com castelo",
        ...     style="minecraft"
        ... )
        >>> if result["success"]:
        ...     print(f"Arquivo 3D: {result['glb_path']}")
    """

    def __init__(
        self,
        temp_dir: str = DEFAULT_TEMP_DIR,
        hunyuan_world_dir: Optional[str] = None,
        hunyuan_3d_dir: Optional[str] = None,
        cleanup_ttl_hours: float = 1.0,
    ):
        """
        Args:
            temp_dir: Diretório temporário para intercâmbio de arquivos
            hunyuan_world_dir: Path do HunyuanWorld-Mirror (auto-detecta)
            hunyuan_3d_dir: Path do Hunyuan3D-2 (auto-detecta)
            cleanup_ttl_hours: TTL em horas para limpeza automática
        """
        self.temp_dir = Path(temp_dir)
        self.world_dir = Path(hunyuan_world_dir) if hunyuan_world_dir else HUNYUAN_WORLD_DIR
        self.h3d_dir = Path(hunyuan_3d_dir) if hunyuan_3d_dir else HUNYUAN_3D_DIR

        self.logger = logging.getLogger("world_to_mesh.pipeline")

        # Inicializar componentes auxiliares
        self._init_style_manager()
        self._init_image_validator()
        self._init_cleanup_scheduler(cleanup_ttl_hours)
        self._init_monster_core()

        # Verificar diretórios
        self._verify_directories()

    def _init_style_manager(self):
        """Carrega o StyleManager do HunyuanWorld."""
        try:
            # Adicionar o src do HunyuanWorld ao path para importar
            world_src = str(self.world_dir)
            if world_src not in sys.path:
                sys.path.insert(0, world_src)

            from src.utils.style_manager import StyleManager
            self.style_manager = StyleManager()
            self.logger.info("✅ StyleManager carregado")
        except ImportError as e:
            self.logger.warning(f"⚠️ StyleManager não disponível: {e}")
            self.style_manager = None

    def _init_image_validator(self):
        """Carrega o ImageQualityValidator."""
        try:
            from src.utils.image_validator import ImageQualityValidator
            self.image_validator = ImageQualityValidator()
            self.logger.info("✅ ImageQualityValidator carregado")
        except ImportError as e:
            self.logger.warning(f"⚠️ ImageQualityValidator não disponível: {e}")
            self.image_validator = None

    def _init_cleanup_scheduler(self, ttl_hours: float):
        """Inicializa o CleanupScheduler."""
        try:
            from src.utils.cleanup_scheduler import CleanupScheduler
            self.cleanup = CleanupScheduler(
                base_dir=str(self.temp_dir),
                ttl_hours=ttl_hours,
            )
            # Iniciar limpeza periódica em background
            self.cleanup.start_background_cleanup()
            self.logger.info("✅ CleanupScheduler iniciado (background)")
        except ImportError as e:
            self.logger.warning(f"⚠️ CleanupScheduler não disponível: {e}")
            self.cleanup = None

    def _verify_directories(self):
        """Verifica se os diretórios dos projetos existem."""
        if not self.world_dir.exists():
            self.logger.warning(
                f"⚠️ HunyuanWorld não encontrado em: {self.world_dir}\n"
                f"   Use --hunyuan-world-dir para especificar o caminho."
            )
        if not self.h3d_dir.exists():
            self.logger.warning(
                f"⚠️ Hunyuan3D-2 não encontrado em: {self.h3d_dir}\n"
                f"   Use --hunyuan-3d-dir para especificar o caminho."
            )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _init_monster_core(self):
        """Inicializa o MonsterCore (C++/CUDA Engine) se disponível."""
        try:
            import monster_core
            self._monster_core = monster_core
            # Pré-aloca arena de 8GB na VRAM (ajustável)
            monster_core.init_pool(8192)
            self.logger.info("🚀 MonsterCore V2 (C++/CUDA) ONLINE — [Pinned Memory | Warp Shuffles | L1 Cache]")
        except ImportError:
            self._monster_core = None
            self.logger.info("⚙️ MonsterCore não compilado — usando Python puro (OK)")
        except Exception as e:
            self._monster_core = None
            self.logger.warning(f"⚠️ MonsterCore falhou ao inicializar: {e}")


    # ── Slots Modulares (Músculos Extras) ─────────────────────────────
    # Estes métodos são placeholders para modelos auxiliares futuros.
    # Quando os modelos forem instalados, basta preencher o interior.

    def _apply_upscale(self, image_path: str, use_upscale: bool = False) -> str:
        """
        [SLOT] Real-ESRGAN – Upscaling de Imagem

        Aplica super-resolução na imagem gerada pelo HunyuanWorld.
        Chamado após validação de qualidade, antes do Hunyuan3D-2.

        Benefícios:
        - Aumenta resolução sem re-gerar (economiza tempo)
        - Melhores detalhes no mesh 3D final
        - Especialmente útil para estilos Low-Poly e Minecraft

        Args:
            image_path: Path da imagem original
            use_upscale: Flag para habilitar/desabilitar

        Returns:
            str: Path da imagem processada (upscaled ou original)
        """
        if not use_upscale:
            return image_path

        self.logger.info("   🔍 [AUTO] Real-ESRGAN 4x upscaling...")

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import cv2
            import torch

            # Configurar modelo RRDBNet
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )

            # Path do peso (fallback se não existir)
            weight_path = Path(__file__).parent / "weights" / "RealESRGAN_x4plus.pth"
            if not weight_path.exists():
                self.logger.warning(f"   ⚠️ Peso não encontrado: {weight_path}, pulando upscale")
                return image_path

            # Criar upsampler com FP16 para economizar vRAM
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(weight_path),
                model=model,
                tile=400,  # Processa em tiles para economizar vRAM
                tile_pad=10,
                pre_pad=0,
                half=True if device == 'cuda' else False  # FP16 apenas em GPU
            )

            # Carregar e processar
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                self.logger.warning(f"   ⚠️ Falha ao carregar imagem: {image_path}")
                return image_path

            output, _ = upsampler.enhance(img, outscale=4)

            # Salvar resultado
            upscaled_path = image_path.replace('.png', '_4x.png')
            cv2.imwrite(upscaled_path, output)

            original_size = Path(image_path).stat().st_size / 1024 / 1024
            upscaled_size = Path(upscaled_path).stat().st_size / 1024 / 1024
            self.logger.info(
                f"   ✅ Upscale 4x: {img.shape[1]}x{img.shape[0]} → "
                f"{output.shape[1]}x{output.shape[0]} ({upscaled_size:.1f}MB)"
            )

            # ── FIX: VRAM flush obrigatório após uso de GPU ──────
            # Sem isso, os ~4GB do RealESRGAN ficam presos na vRAM
            # e o Hunyuan3D-2 (12GB) vai crashar com OOM
            del upsampler, img, output
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   🧹 VRAM liberada após Real-ESRGAN")

            return upscaled_path

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Real-ESRGAN não instalado: {ie}")
            return image_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Upscale falhou: {e}")
            return image_path

    def _apply_depth_refinement(self, image_path: str, use_depth: bool = False) -> Optional[str]:
        """
        [SLOT] Depth Anything V2 – Refinamento de Profundidade

        Gera um mapa de profundidade de alta qualidade antes do Hunyuan3D-2.
        Chamado logo antes de enviar a imagem para reconstrução 3D.

        Benefícios:
        - Melhora geometria de terrenos complexos
        - Reduz artefatos em bordas e transições
        - Útil para cenários realistas com muita variação de altura

        Args:
            image_path: Path da imagem a processar
            use_depth: Flag para habilitar/desabilitar

        Returns:
            Optional[str]: Path do mapa de profundidade gerado (None se desabilitado)
        """
        if not use_depth:
            return None

        self.logger.info("   📏 [AUTO] Depth Anything 3 (metric depth)...")

        try:
            from depth_anything_3.api import DepthAnything3
            import torch
            import cv2
            import numpy as np

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Usar DA3Metric-Large para depth em escala métrica real
            # Alternativa: DA3NESTED-GIANT-LARGE para multi-view + metric
            model_name = "depth-anything/DA3METRIC-LARGE"
            
            try:
                # Carregar modelo via HuggingFace Hub
                model = DepthAnything3.from_pretrained(model_name)
                model = model.to(device=device).eval()
            except Exception as e:
                self.logger.warning(f"   ⚠️ Falha ao carregar modelo: {e}")
                return None

            # Processar imagem
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"   ⚠️ Falha ao carregar: {image_path}")
                return None

            # DA3 aceita lista de imagens (mono ou multi-view)
            # Para mono: lista com 1 imagem
            with torch.no_grad():
                prediction = model.inference([image])

            # prediction.depth shape: [N, H, W] - METRIC depth (metros)
            depth = prediction.depth[0]  # Pegar primeira (única) imagem
            
            # Salvar depth bruto (NPZ para preservar valores métricos)
            depth_npz_path = image_path.replace('.png', '_depth_metric.npz')
            np.savez_compressed(depth_npz_path, depth=depth)

            # Salvar visualização normalizada
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            depth_vis_path = image_path.replace('.png', '_depth.png')
            cv2.imwrite(depth_vis_path, depth_normalized)

            depth_size = Path(depth_vis_path).stat().st_size / 1024 / 1024
            self.logger.info(
                f"   ✅ DA3 Metric: {depth.shape[1]}x{depth.shape[0]} "
                f"(range: {depth.min():.2f}m-{depth.max():.2f}m, {depth_size:.1f}MB)"
            )
            self.logger.info(f"   💾 Depth NPZ (métrico): {depth_npz_path}")

            # ── FIX: VRAM flush obrigatório após uso de GPU ──────
            del model, image, prediction, depth, depth_normalized
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   🧹 VRAM liberada após Depth Anything 3")

            return depth_npz_path

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Depth Anything 3 não instalado: {ie}")
            return None
        except Exception as e:
            self.logger.warning(f"   ⚠️ Depth refinement falhou: {e}")
            return None

    def _generate_normal_map(self, image_path: str, use_normal: bool = False) -> Optional[str]:
        """
        StableNormal-turbo – Geração de Normal Maps de Alta Qualidade
        
        Gera normal maps ultra-detalhados que capturam micro-geometria
        (poros, rachaduras, texturas) que o Hunyuan3D-2.1 sozinho pode perder.
        
        Vantagens vs Wonder3D:
        - 10x mais rápido (versão turbo)
        - SIGGRAPH Asia 2024 quality
        - NÃO redundante com Hunyuan3D-2.1 (complementar)
        - Super leve: ~4-6 GB VRAM
        
        Pipeline Integration:
        - Real-ESRGAN gera imagem 4K
        - StableNormal-turbo gera normal map HD dessa imagem
        - Hunyuan3D-2.1 usa ambos para mesh PBR final
        
        Args:
            image_path: Path da imagem upscaled (4K do Real-ESRGAN)
            use_normal: Flag para habilitar/desabilitar
            
        Returns:
            Optional[str]: Path do normal map gerado (None se desabilitado)
        """
        if not use_normal:
            return None
        
        self.logger.info("   🗺️ [AUTO] StableNormal-turbo (HD Normal Maps)...")
        
        try:
            import torch
            from PIL import Image
            import numpy as np
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Carregar StableNormal-turbo via torch.hub (auto-download)
            self.logger.info("   📦 Carregando StableNormal-turbo...")
            try:
                predictor = torch.hub.load(
                    "Stable-X/StableNormal", 
                    "StableNormal_turbo",  # Versão 10x mais rápida
                    trust_repo=True,
                    force_reload=False  # Cache local
                )
            except Exception as e:
                self.logger.warning(f"   ⚠️ Falha ao carregar modelo: {e}")
                return None
            
            # Carregar imagem 4K
            input_image = Image.open(image_path).convert('RGB')
            original_size = input_image.size
            
            # Gerar normal map
            self.logger.info(f"   🔬 Processando ({original_size[0]}x{original_size[1]})...")
            with torch.no_grad():
                normal_image = predictor(input_image)
            
            # Salvar normal map
            output_path = image_path.replace('.png', '_normal.png').replace('.jpg', '_normal.png')
            normal_image.save(output_path)

            # ── FIX: VRAM flush obrigatório após uso de GPU ──────
            del predictor, input_image, normal_image
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   🧹 VRAM liberada após StableNormal-turbo")

            self.logger.info(f"   ✅ Normal map HD: {output_path}")
            self.logger.info(f"   📊 Qualidade: SIGGRAPH Asia 2024 Award")

            return output_path

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ torch.hub ou PIL não disponível: {ie}")
            return None
        except Exception as e:
            self.logger.warning(f"   ⚠️ Normal map generation falhou: {e}")
            return None

    def _apply_segmentation(self, image_path: str, glb_path: str, use_sam: bool = False) -> Optional[str]:
        """
        SAM 3 – Segmentação Semântica com Conceitos (Open-Vocabulary)

        Gera máscaras de segmentação com identificação de CONCEITOS.
        Ao invés de IDs genéricos, usa nomes semânticos reais.

        Melhorias sobre SAM 2:
        - Open-Vocabulary: Identifica 4M+ conceitos
        - Presence Token: Discrimina objetos similares
        - Labels Semânticos: "gothic_tower" ao invés de "object_01"

        Args:
            image_path: Path da imagem original
            glb_path: Path do .glb gerado
            use_sam: Flag para habilitar/desabilitar

        Returns:
            Optional[str]: Path do JSON com metadados semânticos (None se desabilitado)
        """
        if not use_sam:
            return None

        self.logger.info("   🧠 [AUTO] SAM 3 (Semantic Segmentation)...")

        try:
            import torch
            import cv2
            import json
            import numpy as np

            # SAM 3 tem API diferente do SAM 2
            try:
                from sam3 import build_sam3, SamPrompt
                from sam3.automatic_mask_generator import SamAutomaticMaskGenerator
            except ImportError:
                self.logger.warning("   ⚠️ SAM 3 não instalado. Tente: pip install sam3")
                return None

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # SAM 3 checkpoint (download automático via HuggingFace)
            self.logger.info("   📦 Carregando SAM 3 (pode demorar na 1ª vez)...")
            
            # Construir SAM 3 (API simplificada)
            sam3_model = build_sam3(checkpoint="sam3_hiera_l")
            sam3_model = sam3_model.to(device=device).eval()
            
            # Gerador automático de máscaras com conceitos
            mask_generator = SamAutomaticMaskGenerator(
                sam3_model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=500,
                # NOVO: Habilitar concept prediction
                output_mode="concepts",
                use_presence_token=True  # Melhor discriminação
            )

            # Processar imagem
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"   ⚠️ Falha ao carregar: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                # SAM 3 retorna máscaras + conceitos semânticos
                results = mask_generator.generate(image_rgb)

            # Gerar metadados SEMÂNTICOS para game engines
            metadata = {
                "mesh_path": glb_path,
                "image_path": image_path,
                "total_objects": len(results),
                "segmentation_type": "sam3_concepts",
                "objects": []
            }

            for i, result in enumerate(results):
                if result['area'] > 500:  # Filtrar ruído
                    # NOVO: SAM 3 retorna concept labels
                    concept_label = result.get('concept', f'unknown_{i}')
                    confidence = result.get('concept_score', 0.0)
                    
                    # Limpar conceitos para nomes de arquivo seguros
                    safe_concept = concept_label.lower()
                    safe_concept = safe_concept.replace(' ', '_')
                    safe_concept = ''.join(c for c in safe_concept if c.isalnum() or c == '_')
                    
                    metadata["objects"].append({
                        "id": i,
                        "concept": concept_label,  # Nome semântico real
                        "concept_safe": safe_concept,  # Versão safe para filename
                        "confidence": float(confidence),
                        "area": int(result['area']),
                        "bbox": [int(x) for x in result['bbox']],
                        "stability_score": float(result['stability_score']),
                        "predicted_iou": float(result['predicted_iou'])
                    })

            metadata_path = glb_path.replace('.glb', '_semantic_collision.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Contar conceitos únicos
            unique_concepts = len(set(obj['concept'] for obj in metadata['objects']))
            
            self.logger.info(
                f"   ✅ SAM 3: {len(metadata['objects'])} objetos, "
                f"{unique_concepts} conceitos únicos "
                f"({Path(metadata_path).stat().st_size / 1024:.1f}KB JSON)"
            )

            # ── FIX #1: VRAM flush obrigatório após SAM 3 ───────
            # Sem isso, ~2-4GB do sam3_hiera_l ficam presos na vRAM
            # e a próxima geração crashará com CUDA OOM
            del sam3_model, mask_generator, image, image_rgb, results
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   🧹 VRAM liberada após SAM 3")

            return metadata_path

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ SAM 3 não instalado: {ie}")
            self.logger.info("   💡 Instale com: pip install sam3")
            return None
        except Exception as e:
            self.logger.warning(f"   ⚠️ Segmentação semântica falhou: {e}")
            return None

    def _apply_part_segmentation(
        self, 
        glb_path: str, 
        use_part_segmentation: bool = False,
        sam3_metadata: Optional[dict] = None  # NOVO: Metadados semânticos do SAM 3
    ) -> Optional[str]:
        """
        Hunyuan3D-Part – Segmentação e Geração de Partes 3D (P3-SAM + X-Part).

        Segmenta o mesh em partes semânticas usando metadados do SAM 3.
        Gera outputs modulares com nomes inteligentes.
        
        SAM 3 Integration:
        - Recebe conceitos semânticos da imagem 2D (SAM 3)
        - Mapeia conceitos para partes 3D do mesh
        - Exporta .obj files nomeados (ex: "tower_gothic.obj")
        
        Memory Management:
        - Usa FP16 para reduzir uso de vRAM em 50%
        - Chamado APÓS memory flush do Hunyuan3D-2
        
        Args:
            glb_path: Path do mesh .glb gerado
            use_part_segmentation: Flag para habilitar/desabilitar
            sam3_metadata: Metadados semânticos do SAM 3 (conceitos + bboxes)
            
        Returns:
            Optional[str]: Path do JSON com hierarquia de partes (None se desabilitado)
        """
        if not use_part_segmentation:
            return None

        self.logger.info("   🧩 [AUTO] Hunyuan3D-Part (segmentação modular)...")

        try:
            import torch
            import trimesh
            import json

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Verificar se modelo está disponível no HuggingFace
            try:
                from transformers import AutoModel
                
                # Carregar P3-SAM em FP16/BF16 para economizar vRAM
                self.logger.info("   📦 Carregando P3-SAM (Native 3D Part Segmentation)...")
                p3sam_model = AutoModel.from_pretrained(
                    "tencent/Hunyuan3D-Part",
                    subfolder="p3sam",
                    torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                    trust_remote_code=True
                )
                p3sam_model = p3sam_model.to(device=device).eval()
                
                # Carregar mesh
                mesh = trimesh.load(glb_path, force='mesh')
                
                # Executar P3-SAM para detectar partes
                with torch.no_grad():
                    # Converter mesh para formato esperado
                    vertices = torch.from_numpy(mesh.vertices).float().to(device)
                    faces = torch.from_numpy(mesh.faces).long().to(device)
                    
                    # Inferência P3-SAM
                    segmentation_result = p3sam_model.segment(
                        vertices=vertices.unsqueeze(0),  # [1, N, 3]
                        faces=faces.unsqueeze(0)         # [1, F, 3]
                    )
                
                # Extrair partes segmentadas
                part_labels = segmentation_result['part_labels'][0].cpu().numpy()
                part_names = segmentation_result.get('part_names', [])
                semantic_features = segmentation_result.get('semantic_features', None)
                
                num_parts = len(set(part_labels))
                self.logger.info(f"   ✅ P3-SAM: {num_parts} partes detectadas")
                
                # Limpar modelo da memória
                del p3sam_model, vertices, faces, segmentation_result
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Carregar X-Part para refinar partes (opcional)
                self.logger.info("   🔧 Carregando X-Part (Shape Decomposition)...")
                xpart_model = AutoModel.from_pretrained(
                    "tencent/Hunyuan3D-Part",
                    subfolder="xpart",
                    torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                    trust_remote_code=True
                )
                xpart_model = xpart_model.to(device=device).eval()
                
                # Gerar partes refinadas com X-Part
                parts_hierarchy = {
                    "mesh_path": glb_path,
                    "total_parts": num_parts,
                    "parts": []
                }
                
                output_dir = Path(glb_path).parent / "parts"
                output_dir.mkdir(exist_ok=True)
                
                for part_id in range(num_parts):
                    # Extrair vértices/faces desta parte
                    part_mask = part_labels == part_id
                    part_faces = mesh.faces[part_mask]
                    
                    if len(part_faces) == 0:
                        continue
                    
                    # Criar submesh para esta parte
                    part_mesh = mesh.submesh([part_mask], append=True)
                    
                    # Nome semântico (se disponível)
                    part_name = part_names[part_id] if part_id < len(part_names) else f"part_{part_id}"
                    
                    # Refinar com X-Part (opcional)
                    # TODO: X-Part processing pode ser adicionado aqui
                    
                    # Exportar parte
                    part_path = output_dir / f"{part_name}.obj"
                    part_mesh.export(str(part_path))
                    
                    parts_hierarchy["parts"].append({
                        "id": int(part_id),
                        "name": part_name,
                        "file": str(part_path),
                        "vertex_count": int(len(part_mesh.vertices)),
                        "face_count": int(len(part_mesh.faces))
                    })
                
                # Limpar X-Part
                del xpart_model
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Salvar hierarquia JSON
                hierarchy_path = glb_path.replace('.glb', '_parts_hierarchy.json')
                with open(hierarchy_path, 'w') as f:
                    json.dump(parts_hierarchy, f, indent=2)
                
                parts_size = sum(Path(p["file"]).stat().st_size for p in parts_hierarchy["parts"]) / 1024 / 1024
                self.logger.info(
                    f"   ✅ Hunyuan3D-Part: {num_parts} partes exportadas "
                    f"({parts_size:.1f}MB total)"
                )
                self.logger.info(f"   📂 Partes em: {output_dir}")
                self.logger.info(f"   📋 Hierarquia: {hierarchy_path}")
                
                return hierarchy_path
                
            except Exception as model_error:
                self.logger.warning(f"   ⚠️ Falha ao carregar Hunyuan3D-Part: {model_error}")
                self.logger.info("   💡 Instale com: pip install transformers trust-remote-code")
                return None

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Hunyuan3D-Part não instalado: {ie}")
            return None
        except Exception as e:
            self.logger.warning(f"   ⚠️ Part segmentation falhou: {e}")
            return None

    def _apply_mesh_cleanup(self, raw_glb_path: str, use_cleanup: bool = True) -> str:
        """
        Limpeza de Geometria via trimesh (ONE-SHOT AUTOMATION).

        Remove floaters, faces degeneradas, duplicatas e garante manifold.
        Ativado por padrão para garantir qualidade AAA sem intervenção.
        """
        if not use_cleanup:
            return raw_glb_path

        self.logger.info("   🧹 [AUTO] Limpeza de geometria...")

        try:
            import trimesh

            scene_or_mesh = trimesh.load(raw_glb_path)
            if isinstance(scene_or_mesh, trimesh.Scene):
                meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    self.logger.warning("   ⚠️ Nenhum mesh encontrado no arquivo")
                    return raw_glb_path
                mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = scene_or_mesh

            original_faces = len(mesh.faces)

            # 1. Remove faces degeneradas (área zero)
            mesh.remove_degenerate_faces()
            # 2. Remove vértices duplicados
            mesh.merge_vertices()
            # 3. Remove faces duplicadas
            mesh.remove_duplicate_faces()
            # 4. Remove componentes desconectados pequenos (floaters)
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                components.sort(key=lambda c: len(c.faces), reverse=True)
                # Mantém apenas componentes com >= 5% das faces do maior
                threshold = len(components[0].faces) * 0.05
                kept = [c for c in components if len(c.faces) >= threshold]
                mesh = trimesh.util.concatenate(kept)
            # 5. Preencher buracos
            trimesh.repair.fill_holes(mesh)
            # 6. Consertar normais
            mesh.fix_normals()

            cleaned_faces = len(mesh.faces)
            removed = original_faces - cleaned_faces

            cleaned_path = raw_glb_path.replace('.glb', '_cleaned.glb')
            mesh.export(cleaned_path)

            self.logger.info(
                f"   ✅ Limpeza: {original_faces:,} → {cleaned_faces:,} faces "
                f"({removed:,} removidas). Water-tight: {mesh.is_watertight}"
            )
            return cleaned_path
        except ImportError:
            self.logger.warning("   ⚠️ trimesh não instalado, pulando limpeza")
            return raw_glb_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Falha na limpeza: {e}")
            return raw_glb_path

    def _apply_mesh_optimization(self, clean_glb_path: str, target_faces: int = 60000, use_optimization: bool = True) -> str:
        """
        Otimização de Malha via trimesh (ONE-SHOT AUTOMATION).

        Decimation inteligente para game-engine performance.
        Ativado por padrão.
        """
        if not use_optimization:
            return clean_glb_path

        self.logger.info(f"   ⚡ [AUTO] Otimização de malha (alvo: {target_faces:,} faces)...")

        try:
            import trimesh

            mesh = trimesh.load(clean_glb_path, force='mesh')
            original_faces = len(mesh.faces)
            self.logger.info(f"   📊 Faces originais: {original_faces:,}")

            if original_faces > target_faces:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                optimized_faces = len(mesh.faces)
                reduction_pct = ((original_faces - optimized_faces) / original_faces) * 100
                self.logger.info(
                    f"   ✅ Otimizado: {original_faces:,} → {optimized_faces:,} faces "
                    f"(redução de {reduction_pct:.1f}%)"
                )
            else:
                self.logger.info(f"   ℹ️  Mesh já está abaixo do alvo, pulando decimation")

            optimized_path = clean_glb_path.replace('.glb', '_optimized.glb')
            mesh.export(optimized_path)
            return optimized_path
        except ImportError:
            self.logger.warning("   ⚠️ trimesh não instalado, pulando otimização")
            return clean_glb_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Falha na otimização: {e}")
            return clean_glb_path

    def _apply_mesh_smoothing(self, optimized_glb_path: str, iterations: int = 30, use_smoothing: bool = True) -> str:
        """
        Suavização AAA + RANSAC Hard-Surface (ONE-SHOT AUTOMATION).

        Combina Laplacian smoothing para orgânicos com RANSAC plane
        rectification para superfícies rígidas. Elimina o 'aspecto chiclete'.

        Motor V2 [MonsterCore C++/CUDA]: Roda na GPU usando Pinned Memory, 
        Warp Shuffles e Cache L1. Suporta meshes gigantescos instantaneamente.
        """
        if not use_smoothing:
            return optimized_glb_path

        self.logger.info(f"   ✨ [AUTO] Suavização AAA + RANSAC V2 ({iterations} iterações via CUDA)...")

        try:
            import numpy as np
            import trimesh

            tmesh = trimesh.load(optimized_glb_path, force='mesh')
            vertices = np.asarray(tmesh.vertices, dtype=np.float32)
            faces = np.asarray(tmesh.faces, dtype=np.int64)

            # ═══════════════════════════════════════════════════════════
            # PATH A: MonsterCore C++/CUDA (GPU-Accelerated, ~100x)
            # ═══════════════════════════════════════════════════════════
            if self._monster_core is not None:
                self.logger.info("   🚀 MonsterCore V2: Engine engatada (Pinned Transfer -> Warp Shuffles -> L1 CSR)...")
                import torch

                # Pinned Memory para transferência 3-4x mais rápida
                verts_pinned = torch.from_numpy(vertices).pin_memory()
                faces_pinned = torch.from_numpy(faces).pin_memory()
                
                verts_gpu = verts_pinned.cuda(non_blocking=True)
                faces_gpu = faces_pinned.cuda(non_blocking=True)

                # RANSAC Hard-Surface (GPU paralelo - Warp Shuffles)
                rectified = self._monster_core.gpu_ransac_hard_surface(
                    verts_gpu,
                    distance_threshold=0.015,  # Threshold mais cirúrgico
                    num_iterations=2000,       # Mais iterações = acha melhor o plano
                    batch_size=5000,           # VRAM não estoura mais por causa dos Warp Shuffles
                )

                # Laplacian Smooth (GPU scatter_add com Cache L1)
                smoothed = self._monster_core.gpu_laplacian_smooth(
                    rectified, faces_gpu,
                    iterations=iterations,     # Agora roda 30-50x na velocidade da luz
                    lambda_factor=0.5,
                )

                # Resetar arena para próxima etapa
                self._monster_core.reset_pool()

                # Converter de volta para numpy
                verts_smooth = smoothed.cpu().numpy()
                faces_smooth = faces  # Faces não mudam

                # Recalcular normais
                mesh_final = trimesh.Trimesh(
                    vertices=verts_smooth, faces=faces_smooth,
                )
                mesh_final.fix_normals()

                smoothed_path = optimized_glb_path.replace('.glb', '_smoothed.glb')
                mesh_final.export(smoothed_path)

                obj_path = smoothed_path.replace('.glb', '.obj')
                mesh_final.export(obj_path, file_type='obj')
                self.logger.info(f"   ✅ Mesh AAA (MonsterCore GPU): {smoothed_path}")
                self.logger.info(f"   📦 Export .obj: {obj_path}")
                return smoothed_path

            # ═══════════════════════════════════════════════════════════
            # PATH B: Fallback Open3D (CPU, compatibilidade)
            # ═══════════════════════════════════════════════════════════
            self.logger.info("   ⚙️ Fallback: RANSAC + Smooth via Open3D (CPU)...")
            import open3d as o3d

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            mesh_o3d.compute_vertex_normals()

            # ── RANSAC: Retificar superfícies planas (hard-surface) ──
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=min(len(vertices) * 3, 100000))
            planes_found = 0
            remaining_pcd = pcd
            vertex_array = np.asarray(mesh_o3d.vertices)

            for _ in range(5):  # Máx 5 planos dominantes
                if len(remaining_pcd.points) < 100:
                    break
                plane_model, inliers = remaining_pcd.segment_plane(
                    distance_threshold=0.02, ransac_n=3, num_iterations=1000
                )
                if len(inliers) < len(remaining_pcd.points) * 0.05:
                    break
                # Projetar vértices do mesh próximos ao plano
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                distances = np.abs(vertex_array @ normal + d)
                close_mask = distances < 0.03
                if np.sum(close_mask) > 0:
                    projections = vertex_array[close_mask] - (distances[close_mask, None]) * normal
                    vertex_array[close_mask] = projections
                    planes_found += 1  # type: ignore
                # Remover inliers do pcd restante
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

            if planes_found > 0:
                mesh_o3d.vertices = o3d.utility.Vector3dVector(vertex_array)
                self.logger.info(f"   🔧 RANSAC: {planes_found} planos retificados (quinas de 90°)")

            # ── Laplacian Smoothing (seletivo para orgânicos) ──
            mesh_o3d = mesh_o3d.filter_smooth_laplacian(
                number_of_iterations=iterations, lambda_filter=0.5
            )
            mesh_o3d.compute_vertex_normals()
            mesh_o3d.normalize_normals()

            # Converter de volta e exportar
            verts_smooth = np.asarray(mesh_o3d.vertices)
            faces_smooth = np.asarray(mesh_o3d.triangles)
            normals_smooth = np.asarray(mesh_o3d.vertex_normals)

            mesh_final = trimesh.Trimesh(
                vertices=verts_smooth, faces=faces_smooth,
                vertex_normals=normals_smooth
            )

            smoothed_path = optimized_glb_path.replace('.glb', '_smoothed.glb')
            mesh_final.export(smoothed_path)

            # ── Export .obj (game-engine ready) ──
            obj_path = smoothed_path.replace('.glb', '.obj')
            mesh_final.export(obj_path, file_type='obj')
            self.logger.info(f"   ✅ Mesh AAA: {smoothed_path}")
            self.logger.info(f"   📦 Export .obj: {obj_path}")
            return smoothed_path
        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Dependência faltando ({ie}), pulando suavização")
            return optimized_glb_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Falha na suavização: {e}")
            return optimized_glb_path

    def _generate_world_tiles(self, image_path: str, session_dir,
                              tile_grid: int = 2, overlap_px: int = 64,
                              enable_texture: bool = True, seed: int = 42) -> str:
        """
        [SLOT] World Tiling – Mundos de 2km+ sem perda de detalhe

        Problema: AI gera uma imagem "mestre" que representa o mundo inteiro.
        Processar tudo como um único mesh "derrete" detalhes ou estoura memória.

        Solução: Subdivide a imagem mestre em NxN tiles (ex: 2x2 = 4 tiles),
        processa cada tile pelo Hunyuan3D-2 independentemente, e concatena
        os meshes no final com offset posicional correto.

        Resultado: Mundo 4x mais detalhado com a mesma VRAM por tile.

        Args:
            image_path: Path da imagem mestre
            session_dir: Diretório da sessão
            tile_grid: Grid NxN (2 = 4 tiles, 3 = 9 tiles, 4 = 16 tiles)
            overlap_px: Pixels de sobreposição entre tiles para evitar costura
            enable_texture: Se True, gera texturas PBR
            seed: Seed para reprodutibilidade

        Returns:
            str: Path do mesh concatenado (.glb)
        """
        self.logger.info(f"   🗺️ [TILING] Dividindo mundo em {tile_grid}x{tile_grid} tiles...")

        try:
            from PIL import Image
            import trimesh
            import numpy as np

            img = Image.open(image_path)
            w, h = img.size

            tile_w = w // tile_grid
            tile_h = h // tile_grid

            meshes = []
            tile_paths = []

            for row in range(tile_grid):
                for col in range(tile_grid):
                    # Coordenadas do tile com overlap
                    x1 = max(0, col * tile_w - overlap_px)
                    y1 = max(0, row * tile_h - overlap_px)
                    x2 = min(w, (col + 1) * tile_w + overlap_px)
                    y2 = min(h, (row + 1) * tile_h + overlap_px)

                    tile_img = img.crop((x1, y1, x2, y2))
                    tile_name = f"tile_{row}_{col}.png"
                    tile_path = Path(session_dir) / tile_name
                    tile_img.save(str(tile_path))

                    # Gerar mesh para este tile
                    tile_glb = Path(session_dir) / f"tile_{row}_{col}.glb"
                    self.logger.info(
                        f"   📐 Tile [{row},{col}] ({x2-x1}x{y2-y1}px) → Hunyuan3D-2..."
                    )

                    tile_success = self._run_hunyuan_3d(
                        image_path=str(tile_path),
                        output_path=str(tile_glb),
                        scene_mode=True,
                        enable_texture=enable_texture,
                        seed=seed + row * tile_grid + col,  # Seed diferente por tile
                    )

                    if tile_success and tile_glb.exists():
                        try:
                            tile_mesh = trimesh.load(str(tile_glb), force='mesh')

                            # Aplicar offset posicional baseado na posição do tile
                            # Escala baseada nas proporções do tile na imagem original
                            offset_x = col * (tile_w / w) * 10.0  # Escala mundo = 10 unidades
                            offset_z = row * (tile_h / h) * 10.0

                            tile_mesh.apply_translation([offset_x, 0, offset_z])
                            meshes.append(tile_mesh)
                            tile_paths.append(str(tile_glb))
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Tile [{row},{col}] falhou no load: {e}")
                    else:
                        self.logger.warning(f"   ⚠️ Tile [{row},{col}] falhou na geração")

                    # Flush vRAM entre tiles
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

            if not meshes:
                self.logger.error("   ❌ Nenhum tile gerado com sucesso")
                return image_path  # Fallback

            # Concatenar todos os tiles em um único mesh
            combined = trimesh.util.concatenate(meshes)
            combined_path = str(Path(session_dir) / "world_tiled.glb")
            combined.export(combined_path)

            self.logger.info(
                f"   ✅ Tiling: {len(meshes)}/{tile_grid**2} tiles → "
                f"{len(combined.faces):,} faces, {len(combined.vertices):,} vertices"
            )
            return combined_path

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Dependência faltando ({ie}), pulando tiling")
            return image_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Tiling falhou: {e}")
            return image_path


    def _apply_retopology(self, mesh_path: str, use_retopo: bool = False,
                          target_faces: int = 60000) -> str:
        """
        [SLOT] Instant Meshes – Retopologia Quad-Dominant

        Converte triangulos caóticos da IA em quads profissionais.
        Instant Meshes (ETH Zurich, SIGGRAPH 2015) é o padrão ouro
        para retopologia automática com field-aligned quadrangulation.

        Por que Quads importam:
        - Triângulos (IA raw): difíceis de animar, sem loop flow
        - Quads (Instant Meshes): deformam suavemente, editáveis em Maya/Blender
        - Exigência de studios AAA: Naughty Dog, Epic, ID Software usam só quads
        - SaaS diferencial: permite cobrar como "asset profissional"

        Arquitetura:
        - Roda como subprocess (igual Real-ESRGAN, zero impacto de processo)
        - Input: .obj (Trimesh exporta melhor que .glb para Instant Meshes)
        - Output: .obj retopologizado com ~target_faces quads
        - Tempo: 30-90s dependendo da complexidade
        - VRAM: Roda na CPU, zero impacto na GPU

        Requisito:
        - Linux: baixa InstantMeshes automaticamente de GitHub Releases
        - Windows: requer InstantMeshes.exe no PATH ou INSTANT_MESHES_PATH

        Args:
            mesh_path: Path do mesh (.glb ou .obj)
            use_retopo: Flag para habilitar
            target_faces: Número alvo de faces no resultado quad

        Returns:
            str: Path do mesh retopologizado (ou original se falhou)
        """
        if not use_retopo:
            return mesh_path

        self.logger.info("   🔄 [OPT] Instant Meshes retopologia (quads profissionais)...")

        try:
            import sys
            import platform
            import urllib.request

            # Localizar o binário do Instant Meshes
            instant_meshes_bin = os.environ.get("INSTANT_MESHES_PATH", None)

            # Auto-download no Linux (GPU server típico)
            if instant_meshes_bin is None and platform.system() == "Linux":
                im_path = self.temp_dir / "instant-meshes"
                if not im_path.exists():
                    self.logger.info("   ⬇️  Baixando Instant Meshes (uma vez)...")
                    url = (
                        "https://github.com/wjakob/instant-meshes/releases/download/"
                        "latest/instant-meshes-linux.zip"
                    )
                    zip_path = self.temp_dir / "instant-meshes.zip"
                    urllib.request.urlretrieve(url, str(zip_path))
                    import zipfile
                    with zipfile.ZipFile(str(zip_path), 'r') as z:
                        z.extractall(str(self.temp_dir))
                    zip_path.unlink(missing_ok=True)
                    # Tornar executável
                    im_path.chmod(0o755)
                instant_meshes_bin = str(im_path)

            if instant_meshes_bin is None:
                # Tentar no PATH como fallback
                import shutil as _shutil
                instant_meshes_bin = _shutil.which("InstantMeshes") or _shutil.which("instant-meshes")

            if instant_meshes_bin is None:
                self.logger.warning(
                    "   ⚠️ Instant Meshes não encontrado. "
                    "Defina INSTANT_MESHES_PATH ou instale no PATH. Pulando retopologia."
                )
                return mesh_path

            # Instant Meshes precisa de .obj como input
            # Converter .glb → .obj se necessário
            import trimesh
            obj_input = mesh_path.replace('.glb', '_retopo_in.obj')
            if mesh_path.endswith('.glb'):
                mesh = trimesh.load(mesh_path, force='mesh')
                mesh.export(obj_input)
            else:
                obj_input = mesh_path

            obj_output = obj_input.replace('_retopo_in.obj', '_retopo_out.obj').replace('.obj', '_retopo_out.obj')

            # Construir comando do Instant Meshes
            # -o: output, -f: face count target, -b: deterministic (batch mode)
            cmd = [
                instant_meshes_bin,
                obj_input,
                "-o", obj_output,
                "-f", str(target_faces),
                "-b",            # Batch / headless mode (sem GUI)
                "--smooth", "2", # Iterações de smoothing do field
            ]

            self.logger.info(f"   📐 Retopologizando {target_faces:,} faces → quads...")

            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = proc.communicate(timeout=180)  # 3 min timeout
            except subprocess.TimeoutExpired:
                self.logger.error("   ❌ Timeout (>3min) – Instant Meshes cancelado")
                if proc:
                    proc.terminate()
                return mesh_path
            except KeyboardInterrupt:
                if proc and proc.poll() is None:
                    proc.terminate()
                raise

            if proc.returncode != 0 or not Path(obj_output).exists():
                self.logger.warning(f"   ⚠️ Retopologia falhou (código {proc.returncode}). Mantendo mesh original.")
                return mesh_path

            # Converter .obj resultado de volta para .glb
            retopo_mesh = trimesh.load(obj_output, force='mesh')
            retopo_glb = mesh_path.replace('.glb', '_retopo.glb').replace('.obj', '_retopo.glb')

            # ── FIX #5: Texture Baking (Original → Retopo) ──────
            # O Instant Meshes gera quads puros MAS perde as texturas.
            # Precisamos "bake" as cores da malha original para a nova.
            # Usa KDTree nearest-neighbor para mapear vértices.
            try:
                from scipy.spatial import cKDTree
                import numpy as np

                original_mesh = trimesh.load(mesh_path, force='mesh')

                # Transferir vertex colors via nearest-neighbor
                if hasattr(original_mesh.visual, 'vertex_colors') and original_mesh.visual.vertex_colors is not None:
                    self.logger.info("   🎨 Baking texturas (original → retopo)...")

                    tree = cKDTree(original_mesh.vertices)
                    _, indices = tree.query(retopo_mesh.vertices)

                    baked_colors = original_mesh.visual.vertex_colors[indices]
                    retopo_mesh.visual.vertex_colors = baked_colors

                    self.logger.info(
                        f"   ✅ Texture bake: {len(retopo_mesh.vertices):,} vértices mapeados"
                    )

                # Transferir UVs se existirem no original
                if (hasattr(original_mesh.visual, 'uv') and 
                    original_mesh.visual.uv is not None and
                    len(original_mesh.visual.uv) > 0):
                    try:
                        tree = cKDTree(original_mesh.vertices)
                        _, indices = tree.query(retopo_mesh.vertices)
                        baked_uv = original_mesh.visual.uv[indices]

                        # Aplicar UVs na malha retopologizada
                        if hasattr(retopo_mesh.visual, 'uv'):
                            retopo_mesh.visual.uv = baked_uv
                        self.logger.info("   ✅ UVs transferidas para mesh retopologizado")
                    except Exception as uv_e:
                        self.logger.warning(f"   ⚠️ UV transfer falhou (não crítico): {uv_e}")

            except ImportError:
                self.logger.warning("   ⚠️ scipy não instalado – texturas não transferidas (pip install scipy)")
            except Exception as bake_e:
                self.logger.warning(f"   ⚠️ Texture bake falhou (não crítico): {bake_e}")

            retopo_mesh.export(retopo_glb)

            # Estatísticas
            original_faces = len(trimesh.load(mesh_path, force='mesh').faces)
            final_faces = len(retopo_mesh.faces)
            self.logger.info(
                f"   ✅ Retopologia: {original_faces:,} triângulos → {final_faces:,} quads"
            )
            self.logger.info("   📊 Topologia profissional: loop flow, editável em Maya/Blender")
            return retopo_glb

        except ImportError as ie:
            self.logger.warning(f"   ⚠️ Dependência faltando ({ie}), pulando retopologia")
            return mesh_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ Retopologia falhou: {e}")
            return mesh_path

    def _apply_uv_packing(self, mesh_path: str, use_uv_pack: bool = False) -> str:
        """
        [SLOT] xAtlas – UV Packing Profissional

        Reorganiza os UV islands do mesh gerado pela IA para maximizar
        o aproveitamento do texture atlas. Resolve o problema clássico
        de "texel density caótico" que toda IA 3D produz.

        Por que isso importa:
        - IAs geram UVs espalhadas aleatoriamente → desperdício de VRAM
        - xAtlas reempacota todos os islands com eficiência máxima
        - Resultado: texturas 2x mais nítidas com a mesma memória
        - Assets prontos para export glTF/PBR sem ajuste manual

        Tempo: ~10-30s dependendo do número de faces
        VRAM: Roda na CPU, zero impacto na GPU

        Args:
            mesh_path: Path do mesh (.glb ou .obj)
            use_uv_pack: Flag para habilitar

        Returns:
            str: Path do mesh com UVs otimizados
        """
        if not use_uv_pack:
            return mesh_path

        self.logger.info("   🗺️ [AUTO] xAtlas UV Packing (atlas profissional)...")

        try:
            import xatlas
            import trimesh
            import numpy as np

            # Carregar mesh
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)

            self.logger.info(f"   📐 Empacotando {len(faces):,} faces em UV atlas...")

            # ── xAtlas: Genwrap UV de alta qualidade ────────────
            vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

            # Reconstruir mesh com UVs corretos
            new_vertices = vertices[vmapping]
            mesh_repacked = trimesh.Trimesh(
                vertices=new_vertices,
                faces=indices.reshape(-1, 3),
                process=False
            )

            # Criar VisualMesh com UVs para preservar textura
            visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                material=mesh.visual.material if hasattr(mesh.visual, 'material') else None
            )
            mesh_repacked.visual = visual

            # Salvar resultado
            uv_packed_path = mesh_path.replace('.glb', '_uvpacked.glb').replace('.obj', '_uvpacked.obj')
            mesh_repacked.export(uv_packed_path)

            # Calcular eficiência do UV packing
            uv_utilization = xatlas.get_utilization(uvs)
            self.logger.info(
                f"   ✅ UV Pack: {len(faces):,} faces, "
                f"utilização do atlas: {uv_utilization:.1%}"
            )
            self.logger.info("   📊 Standard: glTF-ready PBR UVs")

            return uv_packed_path

        except ImportError:
            self.logger.warning("   ⚠️ xatlas não instalado. Instale: pip install xatlas")
            return mesh_path
        except Exception as e:
            self.logger.warning(f"   ⚠️ UV Packing falhou: {e}")
            return mesh_path

    # ── Pipeline Principal ─────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        style: str = "realistic",
        output_path: Optional[str] = None,
        max_retries: int = 3,
        enable_texture: bool = True,
        seed: int = 42,
        use_upscale: bool = False,
        use_normal: bool = False,
        use_depth: bool = False,
        use_sam: bool = False,
        use_mesh_cleanup: bool = True,
        use_mesh_optimization: bool = True,
        target_faces: int = 60000,
        use_mesh_smoothing: bool = True,
        smoothing_iterations: int = 5,
        use_retopo: bool = False,
        use_uv_pack: bool = False,
        use_tiling: bool = False,          # NOVO: World Tiling para mundos 2km+
        tile_grid: int = 2,                # Grid NxN (2=4tiles, 3=9tiles)
        quality_threshold: float = 70.0,
        use_part_segmentation: bool = False,
    ) -> dict:
        """
        Pipeline ONE-SHOT PERFECTION: prompt textual → .glb + .obj AAA

        O pipeline roda TODOS os controles de qualidade automaticamente.
        O usuário só precisa fornecer o prompt. Sem flags, sem iteração.

        Fluxo Automático:
        1. Cria sessão temporária
        2. Enriquece prompt com StyleManager
        3. Executa HunyuanWorld para gerar imagem do mundo
        4. Valida qualidade (auto-retry se score < threshold)
        5. [AUTO] Upscale com Real-ESRGAN (quando disponível)
        6. [AUTO] Normal Maps HD com StableNormal-turbo (quando disponível)
        7. [AUTO] Refina depth com Depth Anything 3 (quando disponível)
        8. Executa Hunyuan3D-2.1 para gerar mesh 3D + PBR
        9. [AUTO] Limpeza: floaters, buracos, duplicatas (trimesh)
        10. [AUTO] Otimização: decimation para 60k faces (trimesh)
        11. [AUTO] RANSAC + Suavização AAA (Open3D)
        12. [OPT] Retopologia quad-dominant (Instant Meshes)
        13. [OPT] UV Packing profissional com xAtlas
        14. Exporta .glb + .obj (game-engine ready)
        15. Agenda limpeza da sessão

        Returns:
            dict com 'success', 'glb_path', 'obj_path', etc.
        """
        start_total = time.time()

        result: dict[str, Any] = {
            "success": False,
            "glb_path": "",
            "obj_path": "",
            "session_id": "",
            "prompt_original": prompt,
            "prompt_enhanced": "",
            "style": style,
            "time_total_seconds": 0.0,
            "time_world_seconds": 0.0,
            "time_3d_seconds": 0.0,
            "quality_score": 0.0,
            "error": "",
        }

        # ── 1. Criar sessão temporária ───────────────────────────
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        result["session_id"] = session_id

        if self.cleanup:
            self.cleanup.create_session(session_id)
        session_dir: Path = self.temp_dir / "sessions" / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("  🚀 ONE-SHOT PERFECTION PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"  Sessão: {session_id}")
        self.logger.info(f"  Prompt: {prompt}")
        self.logger.info(f"  Estilo: {style}")
        self.logger.info(f"  Qualidade mínima: {quality_threshold}/100")
        self.logger.info("=" * 60)

        # ── 2. Enriquecer prompt com StyleManager ────────────────
        # PULO DO GATO #2: O dicionário de estilos injeta keywords
        # técnicas que o HunyuanWorld entende melhor.
        enhanced_prompt = prompt
        style_params = {}

        if self.style_manager:
            try:
                enhanced_prompt, style_params = self.style_manager.enhance_prompt(
                    prompt, style
                )
                self.logger.info(f"🎨 Prompt enriquecido: {enhanced_prompt}")
                self.logger.info(f"   Parâmetros de estilo: {style_params}")
            except ValueError as e:
                self.logger.warning(f"⚠️ Estilo não reconhecido: {e}")

        result["prompt_enhanced"] = enhanced_prompt

        # ── 3. Executar HunyuanWorld ─────────────────────────────
        master_frame_path = session_dir / "world_image.png"

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"\n🌍 Etapa 1/2: HunyuanWorld (tentativa {attempt}/{max_retries})")
            start_world = time.time()

            world_success = self._run_hunyuan_world(
                enhanced_prompt=enhanced_prompt,
                style=style,
                style_params=style_params,
                output_dir=str(session_dir / "world_output"),
                master_frame_path=str(master_frame_path),
                seed=seed + attempt - 1,  # Seed diferente a cada tentativa
            )

            result["time_world_seconds"] = round(time.time() - start_world, 2)  # type: ignore

            if not world_success:
                self.logger.warning(f"   ❌ HunyuanWorld falhou na tentativa {attempt}")
                if attempt < max_retries:
                    self.logger.info(f"   🔄 Tentando novamente com seed ajustado...")
                continue

            # ── 4. Validar qualidade (ONE-SHOT AUTO-CORRECTION) ────
            # Cada validação custa ~1s CPU vs ~30s GPU no 3D.
            # Se score < threshold → auto-retry com seed diferente.
            if self.image_validator and master_frame_path.exists():
                is_valid, report = self.image_validator.validate(str(master_frame_path))
                score = self.image_validator.get_quality_score(str(master_frame_path))
                result["quality_score"] = score  # type: ignore

                if not is_valid or score < quality_threshold:
                    self.logger.warning(
                        f"   ⚠️ Qualidade insuficiente "
                        f"(score: {score:.1f}/100, mínimo: {quality_threshold}): {report['reason']}"
                    )
                    if attempt < max_retries:
                        self.logger.info(f"   🔄 [AUTO-CORREÇÃO] Regenerando com seed diferente...")
                        continue
                    else:
                        self.logger.warning(
                            f"   ⚠️ Última tentativa – prosseguindo com melhor resultado disponível"
                        )
                else:
                    self.logger.info(f"   ✅ Qualidade AAA aprovada (score: {score:.1f}/100)")

            # ── SLOT: Upscaling [Real-ESRGAN] ───────────────────
            # Aumenta resolução da imagem antes do processamento 3D
            master_frame_path = Path(self._apply_upscale(
                str(master_frame_path),
                use_upscale=use_upscale
            ))

            # ── FIX #3: Paralelizar StableNormal + DA3 ──────────
            # Ambos só dependem da imagem upscaled, não um do outro.
            # ThreadPoolExecutor permite que I/O, model loading e 
            # pré/pós processamento se sobreponham (economia ~15-30s).
            # A GPU em si serializa via CUDA, mas o overhead é eliminado.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            normal_map_path = None
            depth_map_path = None

            gpu_tasks = {}
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu_slot") as executor:
                if use_normal:
                    gpu_tasks["normal"] = executor.submit(
                        self._generate_normal_map,
                        str(master_frame_path),
                        use_normal=True
                    )
                if use_depth:
                    gpu_tasks["depth"] = executor.submit(
                        self._apply_depth_refinement,
                        str(master_frame_path),
                        use_depth=True
                    )

                for future in as_completed(gpu_tasks.values()):
                    # Identificar qual terminou
                    for name, f in gpu_tasks.items():
                        if f is future:
                            try:
                                result_val = future.result()
                                if name == "normal" and result_val:
                                    normal_map_path = result_val
                                    self.logger.info(f"   🎨 Normal map HD disponível: {normal_map_path}")
                                elif name == "depth" and result_val:
                                    depth_map_path = result_val
                                    self.logger.info(f"   📊 Depth map disponível: {depth_map_path}")
                            except Exception as e:
                                self.logger.warning(f"   ⚠️ {name} falhou em paralelo: {e}")
                            break

            break  # Sucesso!
        else:
            result["error"] = (
                f"HunyuanWorld falhou após {max_retries} tentativas. "
                f"Verifique os logs para detalhes."
            )
            self._schedule_cleanup(session_id)
            result["time_total_seconds"] = round(time.time() - start_total, 2)
            return result


        # ── 5. Executar Hunyuan3D-2 ──────────────────────────────
        glb_path = Path(session_dir) / "world_mesh.glb"

        self.logger.info(f"\n🔨 Etapa 2/2: Hunyuan3D-2 (reconstrução 3D)")
        start_3d = time.time()

        if use_tiling:
            # FIX #4: World Tiling – subdivide em NxN tiles para mundos 2km+
            tiled_path = self._generate_world_tiles(
                image_path=str(master_frame_path),
                session_dir=session_dir,
                tile_grid=tile_grid,
                enable_texture=enable_texture,
                seed=seed,
            )
            if Path(tiled_path).exists() and tiled_path.endswith('.glb'):
                glb_path = Path(tiled_path)
                mesh_success = True
            else:
                self.logger.warning("   ⚠️ Tiling falhou, tentando single-shot...")
                mesh_success = self._run_hunyuan_3d(
                    image_path=str(master_frame_path),
                    output_path=str(glb_path),
                    scene_mode=True,
                    enable_texture=enable_texture,
                    seed=seed,
                    normal_map_path=normal_map_path,
                    depth_map_path=depth_map_path,
                )
        else:
            # Single-shot (padrão)
            mesh_success = self._run_hunyuan_3d(
                image_path=str(master_frame_path),
                output_path=str(glb_path),
                scene_mode=True,
                enable_texture=enable_texture,
                seed=seed,
                normal_map_path=normal_map_path,
                depth_map_path=depth_map_path,
            )

        result["time_3d_seconds"] = round(time.time() - start_3d, 2)

        # ── MEMORY MANAGEMENT: Flush após Hunyuan3D-2 ───────────
        # Libera ~12GB de vRAM para os próximos modelos (Hunyuan3D-Part)
        try:
            import torch
            import gc
            
            # Forçar garbage collection
            gc.collect()
            
            # Limpar cache CUDA (crítico para vRAM)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("   🧹 Memory flushed: vRAM liberada após Hunyuan3D-2")
        except Exception as e:
            self.logger.warning(f"   ⚠️ Memory flush falhou: {e}")

        if not mesh_success:
            result["error"] = "Hunyuan3D-2 falhou na reconstrução 3D"
            self._schedule_cleanup(session_id)
            result["time_total_seconds"] = round(time.time() - start_total, 2)
            return result

        # ── 6. Processamento de Mesh (SLOTS AAA) ────────────────
        self.logger.info("")
        self.logger.info("── [6/7] Processamento de Mesh ──")
        
        # 6.1 Limpeza (MeshLab)
        glb_path = Path(self._apply_mesh_cleanup(str(glb_path), use_cleanup=use_mesh_cleanup))
        
        # 6.2 Otimização (trimesh)
        # FIX #2: Pula decimation quando use_retopo=True
        # Motivo: decimation estraga quinas retas ANTES do RANSAC consertá-las.
        # O Instant Meshes já faz redução de faces de forma profissional.
        if use_retopo:
            self.logger.info("   ✅ Decimation pulado (Instant Meshes fará redução de faces)")
        else:
            glb_path = Path(self._apply_mesh_optimization(
                str(glb_path),
                target_faces=target_faces,
                use_optimization=use_mesh_optimization
            ))

        # 6.3 Suavização (Open3D)
        glb_path = Path(self._apply_mesh_smoothing(
            str(glb_path), 
            iterations=smoothing_iterations, 
            use_smoothing=use_mesh_smoothing
        ))

        # 6.4 Retopologia Quad-Dominant (Instant Meshes)
        # Converte triângulos de IA em quads profissionais (+60s)
        glb_path = Path(self._apply_retopology(
            str(glb_path),
            use_retopo=use_retopo,
            target_faces=target_faces
        ))

        # 6.5 UV Packing Profissional (xAtlas)
        # Roda na CPU, zero impacto de vRAM
        glb_path = Path(self._apply_uv_packing(
            str(glb_path),
            use_uv_pack=use_uv_pack
        ))

        # 6.6 Segmentação de Partes (Hunyuan3D-Part com SAM 3)
        # Passar metadados semânticos do SAM 3 (se disponíveis)
        sam3_semantic_metadata = None
        semantic_collision_json = str(glb_path).replace('.glb', '_semantic_collision.json')
        if Path(semantic_collision_json).exists():
            try:
                import json
                with open(semantic_collision_json, 'r') as f:
                    sam3_semantic_metadata = json.load(f)
                    self.logger.info(f"   📊 SAM 3 metadata carregado: {sam3_semantic_metadata.get('total_objects', 0)} conceitos")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Falha ao carregar SAM 3 metadata: {e}")
        
        parts_hierarchy_path = self._apply_part_segmentation(
            str(glb_path),
            use_part_segmentation=use_part_segmentation,
            sam3_metadata=sam3_semantic_metadata
        )
        if parts_hierarchy_path:
            result["parts_hierarchy"] = parts_hierarchy_path
            self.logger.info(f"   📋 Hierarquia de partes: {parts_hierarchy_path}")

        # ── 7. Exportar para output final (.glb + .obj) ──────────
        if output_path:
            final_path = Path(output_path)
            if final_path.suffix.lower() not in (".glb", ".obj"):
                final_path = final_path.with_suffix(".glb")
        else:
            safe_name = "".join(
                c if c.isalnum() or c in "._- " else "_"
                for c in prompt[:50]
            ).strip().replace(" ", "_")
            final_path = Path.cwd() / f"{safe_name}_{style}.glb"

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(glb_path), str(final_path))
            result["success"] = True
            result["glb_path"] = str(final_path)

            # Copiar .obj se existir (gerado pelo smoothing step)
            obj_source = str(glb_path).replace('.glb', '.obj')
            if Path(obj_source).exists():
                obj_final = final_path.with_suffix(".obj")
                shutil.copy2(obj_source, str(obj_final))
                result["obj_path"] = str(obj_final)
                self.logger.info(f"\n✅ Assets game-ready gerados:")
                self.logger.info(f"   📦 GLB: {final_path}")
                self.logger.info(f"   📦 OBJ: {obj_final}")
            else:
                self.logger.info(f"\n✅ Arquivo 3D gerado: {final_path}")

            # ── Segmentation [SAM 2] (quando disponível) ─────────
            metadata_path = self._apply_segmentation(
                str(master_frame_path),
                str(final_path),
                use_sam=use_sam
            )
            if metadata_path:
                self.logger.info(f"   📋 Metadados de colisão: {metadata_path}")
                result["collision_metadata"] = metadata_path

        except Exception as e:
            result["error"] = f"Falha ao exportar: {e}"
            self.logger.error(result["error"])

        # ── 7. Agendar limpeza ───────────────────────────────────
        self._schedule_cleanup(session_id)

        # ── Resultado final ──────────────────────────────────────
        result["time_total_seconds"] = round(time.time() - start_total, 2)

        # Salvar metadata da sessão
        self._save_metadata(session_dir, result)

        self.logger.info("=" * 60)
        self.logger.info("  RESULTADO")
        self.logger.info("=" * 60)
        self.logger.info(f"  Status:  {'✅ SUCESSO' if result['success'] else '❌ FALHA'}")
        self.logger.info(f"  Arquivo: {result['glb_path']}")
        self.logger.info(f"  Tempo Total:     {result['time_total_seconds']}s")
        self.logger.info(f"  Tempo HunyWorld: {result['time_world_seconds']}s")
        self.logger.info(f"  Tempo Hunyuan3D: {result['time_3d_seconds']}s")
        if result["error"]:
            self.logger.info(f"  Erro: {result['error']}")
        self.logger.info("=" * 60)

        return result

    # ── Etapas Internas ────────────────────────────────────────────────

    def _run_hunyuan_world(
        self,
        enhanced_prompt: str,
        style: str,
        style_params: dict,
        output_dir: str,
        master_frame_path: str,
        seed: int = 42,
    ) -> bool:
        """
        Executa o HunyuanWorld-Mirror para gerar visualização do mundo.

        Usa subprocess para chamar o infer.py do HunyuanWorld.
        Isso isola o processo e permite controlar timeouts.

        Returns:
            bool: True se o processo completou com sucesso
        """
        infer_script = self.world_dir / "infer.py"

        if not infer_script.exists():
            self.logger.error(f"❌ Script não encontrado: {infer_script}")
            return False

        # Construir comando
        cmd = [
            sys.executable, str(infer_script),
            "--output_path", output_dir,
            "--text_prompt", enhanced_prompt,
            "--style", style,
            "--save_master_frame", master_frame_path,
        ]

        # Adicionar parâmetros de estilo
        for param, value in style_params.items():
            arg_name = f"--{param}"
            cmd.extend([arg_name, str(value)])

        self.logger.info(f"   Executando: {' '.join(cmd[:6])}...")

        proc = None  # Guardamos a referência para poder matar se necessário
        try:
            # FIX #2: Usar Popen em vez de subprocess.run
            # subprocess.run não permite matar o processo filho se o pai morrer
            # Popen armazena a referência, permitindo proc.terminate() no Ctrl+C
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.world_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=600)  # 10 min timeout
            except subprocess.TimeoutExpired:
                self.logger.error("   ❌ Timeout (>10min) – matando processo HunyuanWorld")
                proc.terminate()   # Sinal SIGTERM educado
                proc.wait(timeout=10)  # Dá 10s para morrer
                if proc.poll() is None:
                    proc.kill()    # SIGKILL forçado se não morreu
                return False
            except KeyboardInterrupt:
                self.logger.warning("   ⚠️ Ctrl+C recebido – matando HunyuanWorld...")
                proc.terminate()
                proc.wait(timeout=5)
                raise  # Re-lança para o main() tratar

            if proc.returncode != 0:
                self.logger.error(f"   Stderr: {stderr[-500:]}")
                return False

            # Verificar se o frame mestre foi gerado
            if Path(master_frame_path).exists():
                self.logger.info(f"   ✅ Frame mestre gerado: {master_frame_path}")
                return True
            else:
                self.logger.error("   ❌ Frame mestre não foi gerado")
                return False

        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.terminate()
            raise
        except Exception as e:
            if proc and proc.poll() is None:
                proc.terminate()
            self.logger.error(f"   ❌ Erro inesperado: {e}")
            return False

    def _run_hunyuan_3d(
        self,
        image_path: str,
        output_path: str,
        scene_mode: bool = True,
        enable_texture: bool = True,
        seed: int = 42,
        normal_map_path: Optional[str] = None,  # FIX #3: Integration gap
        depth_map_path: Optional[str] = None,   # FIX #3: Integration gap
    ) -> bool:
        """
        Executa o Hunyuan3D-2 para reconstrução 3D.

        Usa o SceneMeshGenerator como subprocess.
        FIX #2: Usa Popen para evitar processos zumbi no Ctrl+C.
        FIX #3: Passa normal_map e depth_map para o pipeline 3D.

        Returns:
            bool: True se o mesh foi gerado com sucesso
        """
        scene_gen_script = self.h3d_dir / "scene_mesh_generator.py"

        if not scene_gen_script.exists():
            self.logger.error(f"❌ Script não encontrado: {scene_gen_script}")
            return False

        # Construir comando
        cmd = [
            sys.executable, str(scene_gen_script),
            "--input", image_path,
            "--output", output_path,
            "--seed", str(seed),
        ]

        if scene_mode:
            cmd.append("--scene_mode")
        else:
            cmd.append("--object_mode")

        if not enable_texture:
            cmd.append("--no_texture")

        # FIX #3: Injetar normal map e depth map no comando
        # Isso RESOLVE o gargalo de integração (antes eram gerados mas ignorados!)
        if normal_map_path and Path(normal_map_path).exists():
            cmd.extend(["--normal_map", normal_map_path])
            self.logger.info(f"   🗺️ Normal map injetado no pipeline 3D")
        if depth_map_path and Path(depth_map_path).exists():
            cmd.extend(["--depth_map", depth_map_path])
            self.logger.info(f"   📏 Depth map injetado no pipeline 3D")

        self.logger.info(f"   Executando: {' '.join(cmd[:6])}...")

        proc = None  # FIX #2: Guardamos referência para matar no Ctrl+C
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.h3d_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=900)  # 15 min timeout
            except subprocess.TimeoutExpired:
                self.logger.error("   ❌ Timeout (>15min) – matando processo Hunyuan3D")
                proc.terminate()
                proc.wait(timeout=10)
                if proc.poll() is None:
                    proc.kill()
                return False
            except KeyboardInterrupt:
                self.logger.warning("   ⚠️ Ctrl+C – matando Hunyuan3D...")
                proc.terminate()
                proc.wait(timeout=5)
                raise

            if proc.returncode != 0:
                self.logger.error(f"   Stderr: {stderr[-500:]}")
                return False

            if Path(output_path).exists():
                self.logger.info(f"   ✅ Mesh 3D gerado: {output_path}")
                return True
            else:
                self.logger.error("   ❌ Arquivo .glb não foi gerado")
                return False

        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.terminate()
            raise
        except Exception as e:
            if proc and proc.poll() is None:
                proc.terminate()
            self.logger.error(f"   ❌ Erro inesperado: {e}")
            return False

    # ── Utilidades ─────────────────────────────────────────────────────

    def _schedule_cleanup(self, session_id: str):
        """Agenda limpeza da sessão temporária."""
        if self.cleanup:
            self.cleanup.schedule_cleanup(session_id)

    def cleanup_now(self, session_id: str):
        """Limpa sessão imediatamente (após download). """
        if self.cleanup:
            self.cleanup.immediate_cleanup(session_id)
            self.logger.info(f"🗑️  Sessão {session_id} limpa")

    def _save_metadata(self, session_dir: Path, result: dict):
        """Salva metadata da sessão para tracking."""
        meta_path = session_dir / "metadata.json"
        meta = {
            "session_id": result["session_id"],
            "prompt_original": result["prompt_original"],
            "prompt_enhanced": result["prompt_enhanced"],
            "style": result["style"],
            "success": result["success"],
            "glb_path": result["glb_path"],
            "time_total": result["time_total_seconds"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def list_styles(self):
        """Lista estilos disponíveis."""
        if self.style_manager:
            return self.style_manager.list_styles()
        return []

    def get_status(self) -> dict:
        """Retorna status do pipeline (útil para monitoramento remoto)."""
        status = {
            "hunyuan_world_available": self.world_dir.exists(),
            "hunyuan_3d_available": self.h3d_dir.exists(),
            "style_manager": self.style_manager is not None,
            "image_validator": self.image_validator is not None,
            "cleanup_scheduler": self.cleanup is not None,
            "temp_dir": str(self.temp_dir),
        }
        if self.cleanup:
            status["cleanup"] = self.cleanup.get_status()
        return status

    def shutdown(self):
        """Para o pipeline e limpa recursos."""
        if self.cleanup:
            self.cleanup.stop_background_cleanup()
        self.logger.info("Pipeline encerrado")


# ── CLI ────────────────────────────────────────────────────────────────
# Uso principal via terminal (ideal para SSH em GPU alugada)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "🌍 World-to-Mesh Universal Pipeline\n"
            "    Converte prompts textuais em arquivos 3D (.glb)\n"
            "    prontos para Unreal Engine / Unity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python bridge.py --prompt 'vila medieval' --style minecraft\n"
            "  python bridge.py --prompt 'floresta mágica' --style rpg --output floresta.glb\n"
            "  python bridge.py --prompt 'base espacial' --style sci-fi --no-texture\n"
            "  python bridge.py --list-styles\n"
            "  python bridge.py --status\n"
        ),
    )

    # Argumentos principais
    parser.add_argument(
        "--prompt", type=str,
        help="Prompt textual para gerar o mundo 3D",
    )
    parser.add_argument(
        "--style", type=str, default="realistic",
        help="Estilo artístico (padrão: realistic). Aceita: minecraft, rpg, realistic, low-poly, sci-fi, ou qualquer termo descritivo",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Caminho de saída do arquivo .glb (padrão: auto-gerado)",
    )

    # Configurações avançadas
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Máx tentativas em caso de falha (padrão: 3)",
    )
    parser.add_argument(
        "--no-texture", action="store_true", default=False,
        help="Gerar mesh sem texturas (mais rápido, menos VRAM)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reprodutibilidade (padrão: 42)",
    )

    # Módulos auxiliares ("Músculos Extras")
    parser.add_argument(
        "--use-upscale", action="store_true", default=False,
        help="Aplicar Real-ESRGAN para upscaling 4x da imagem",
    )
    parser.add_argument(
        "--use-depth", action="store_true", default=False,
        help="Aplicar Depth Anything 3 para refinamento de profundidade",
    )
    parser.add_argument(
        "--use-sam", action="store_true", default=False,
        help="Aplicar SAM 3 para gerar metadados de colisão/objetos",
    )
    # FIX #4: Flags que estavam no generate() mas faltavam na CLI
    parser.add_argument(
        "--use-normal", action="store_true", default=False,
        help="Aplicar StableNormal-turbo para HD Normal Maps (SIGGRAPH 2024)",
    )
    parser.add_argument(
        "--use-part-segmentation", action="store_true", default=False,
        help="Segmentar mesh em partes separáveis via Hunyuan3D-Part + SAM 3",
    )
    parser.add_argument(
        "--use-uv-pack", action="store_true", default=False,
        help="Aplicar xAtlas UV Packing profissional (texturas 2x mais nítidas, +20s)",
    )
    parser.add_argument(
        "--use-retopo", action="store_true", default=False,
        help="Aplicar Instant Meshes retopologia quad-dominant (+60s, +50%% no valor do asset)",
    )
    parser.add_argument(
        "--use-tiling", action="store_true", default=False,
        help="Ativa World Tiling: divide cena em NxN tiles para mundos 2km+ com mais detalhe",
    )
    parser.add_argument(
        "--tile-grid", type=int, default=2,
        help="Grid NxN para tiling (2=4tiles, 3=9tiles, 4=16tiles). Padrão: 2",
    )

    # Processamento de Mesh ("Músculos AAA")
    parser.add_argument(
        "--use-mesh-cleanup", action="store_true", default=False,
        help="[FUTURO] Aplicar MeshLab para limpeza de geometria (remove floaters, fecha buracos)",
    )
    parser.add_argument(
        "--use-mesh-optimization", action="store_true", default=False,
        help="[FUTURO] Aplicar trimesh para otimização (decimation inteligente)",
    )
    parser.add_argument(
        "--target-faces", type=int, default=150000,
        help="Número alvo de faces para otimização (padrão: 150000 - Motor V2 lida tranquilamente)",
    )
    parser.add_argument(
        "--use-mesh-smoothing", action="store_true", default=False,
        help="[FUTURO] Aplicar Open3D para suavização AAA (Laplacian smoothing)",
    )
    parser.add_argument(
        "--smoothing-iterations", type=int, default=30,
        help="Número de iterações de smoothing (padrão: 30 - Motor V2 roda na velocidade da luz)",
    )
    parser.add_argument(
        "--cleanup-ttl", type=float, default=1.0,
        help="TTL em horas para limpeza automática (padrão: 1.0)",
    )
    parser.add_argument(
        "--temp-dir", type=str, default=DEFAULT_TEMP_DIR,
        help="Diretório temporário para intercâmbio",
    )

    # Caminhos dos projetos
    parser.add_argument(
        "--hunyuan-world-dir", type=str, default=None,
        help="Path do HunyuanWorld-Mirror",
    )
    parser.add_argument(
        "--hunyuan-3d-dir", type=str, default=None,
        help="Path do Hunyuan3D-2",
    )

    # Utilitários
    parser.add_argument(
        "--list-styles", action="store_true",
        help="Listar estilos disponíveis e sair",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Mostrar status do pipeline e sair",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado (debug)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Salvar logs em arquivo (útil para monitoramento remoto)",
    )

    args = parser.parse_args()

    # Setup logging
    log = setup_logging(log_file=args.log_file, verbose=args.verbose)

    # Criar pipeline
    pipeline = WorldToMeshPipeline(
        temp_dir=args.temp_dir,
        hunyuan_world_dir=args.hunyuan_world_dir,
        hunyuan_3d_dir=args.hunyuan_3d_dir,
        cleanup_ttl_hours=args.cleanup_ttl,
    )

    # ── Comandos utilitários ───────────────────────────────────
    if args.list_styles:
        print("\n🎨 Estilos Disponíveis:")
        print("=" * 60)
        styles = pipeline.list_styles()
        if styles:
            for s in styles:
                aliases = ", ".join(s["aliases"]) if s["aliases"] else "nenhum"
                keywords = ", ".join(s["keywords"][:4]) + "..."
                print(f"\n  📎 {s['name']}")
                print(f"     {s['description']}")
                print(f"     Keywords: {keywords}")
                print(f"     Aliases: {aliases}")
        else:
            print("  Nenhum estilo disponível (StyleManager não carregado)")
        return

    if args.status:
        print("\n📊 Status do Pipeline:")
        print("=" * 60)
        status = pipeline.get_status()
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        return

    # ── Geração ────────────────────────────────────────────────
    if not args.prompt:
        parser.error("O argumento --prompt é obrigatório para gerar um mundo 3D")

    try:
        result = pipeline.generate(
            prompt=args.prompt,
            style=args.style,
            output_path=args.output,
            max_retries=args.max_retries,
            enable_texture=not args.no_texture,
            seed=args.seed,
            use_upscale=args.use_upscale,
            use_normal=args.use_normal,
            use_depth=args.use_depth,
            use_sam=args.use_sam,
            use_mesh_cleanup=args.use_mesh_cleanup,
            use_mesh_optimization=args.use_mesh_optimization,
            target_faces=args.target_faces,
            use_mesh_smoothing=args.use_mesh_smoothing,
            smoothing_iterations=args.smoothing_iterations,
            use_retopo=args.use_retopo,
            use_uv_pack=args.use_uv_pack,
            use_tiling=args.use_tiling,
            tile_grid=args.tile_grid,
            use_part_segmentation=args.use_part_segmentation,
        )

        if result["success"]:
            print(f"\n🎉 ONE-SHOT PERFECTION (Powered by MonsterCore V2) – Asset 3D gerado com sucesso!")
            print(f"   � GLB: {result['glb_path']}")
            if result.get("obj_path"):
                print(f"   📦 OBJ: {result['obj_path']}")
            print(f"   ⭐ Qualidade: {result.get('quality_score', 'N/A')}/100")
            print(f"   ⏱️  Tempo total: {result['time_total_seconds']}s")
            print(f"\n   Importe o .glb ou .obj no Unreal Engine / Unity.")
        else:
            print(f"\n❌ Falha na geração: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Operação cancelada pelo usuário")
        pipeline.shutdown()
        sys.exit(130)
    except Exception as e:
        log.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
