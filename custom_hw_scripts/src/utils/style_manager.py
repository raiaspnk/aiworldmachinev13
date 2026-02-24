"""
==========================================================================
  STYLE MANAGER – Gerenciador de Estilos para World-to-Mesh Pipeline
==========================================================================

Módulo responsável por converter prompts de usuário em prompts técnicos
otimizados para o HunyuanWorld-Mirror. Cada estilo tem um dicionário de
keywords que são injetadas automaticamente no prompt.

Uso via terminal (cloud/GPU alugada):
    from src.utils.style_manager import StyleManager
    sm = StyleManager()
    enhanced, params = sm.enhance_prompt("vila medieval", "minecraft")

Adicionar novos estilos:
    Basta adicionar uma nova entrada em STYLE_PRESETS com:
    - "keywords": lista de termos técnicos
    - "negative_keywords": termos a evitar
    - "params": parâmetros de inferência ajustados
    - "description": descrição legível do estilo
==========================================================================
"""

import logging
from typing import Optional

logger = logging.getLogger("world_to_mesh.style_manager")


# ── Dicionário de Estilos ──────────────────────────────────────────────
# Cada estilo contém keywords técnicas que o HunyuanWorld entende melhor,
# parâmetros de inferência otimizados, e keywords negativas para evitar
# resultados indesejados.

STYLE_PRESETS: dict = {
    "minecraft": {
        "keywords": [
            "voxel",
            "blocky",
            "8-bit texture",
            "cubic geometry",
            "pixelated",
            "block-based world",
            "low-resolution textures",
            "flat lighting",
        ],
        "negative_keywords": [
            "smooth",
            "organic curves",
            "photorealistic",
            "high-poly",
        ],
        "params": {
            "target_size": 518,  # Múltiplo de 14 obrigatório (37 * 14 = 518)
            "edge_normal_threshold": 8.0,   # Arestas bem pronunciadas
            "edge_depth_threshold": 0.05,
            "confidence_percentile": 12.0,
            "fps": 2,
        },
        "description": "Estilo voxel blocky inspirado em Minecraft",
    },

    "rpg": {
        "keywords": [
            "stylized",
            "high-fantasy",
            "hand-painted texture",
            "vibrant colors",
            "medieval fantasy",
            "epic landscape",
            "dramatic lighting",
            "rich detail",
        ],
        "negative_keywords": [
            "modern",
            "sci-fi",
            "minimalist",
            "pixelated",
        ],
        "params": {
            "target_size": 518,
            "edge_normal_threshold": 5.0,
            "edge_depth_threshold": 0.03,
            "confidence_percentile": 15.0,
            "fps": 1,
        },
        "description": "Estilo fantasia RPG com texturas hand-painted",
    },

    "realistic": {
        "keywords": [
            "photorealistic",
            "high-detail",
            "pbr materials",
            "natural lighting",
            "physically based rendering",
            "detailed geometry",
            "atmospheric perspective",
            "volumetric lighting",
        ],
        "negative_keywords": [
            "cartoon",
            "stylized",
            "low-poly",
            "pixelated",
        ],
        "params": {
            "target_size": 770,  # Múltiplo de 14 obrigatório (55 * 14 = 770)
            "edge_normal_threshold": 3.0,   # Mais fino para detalhes
            "edge_depth_threshold": 0.02,
            "confidence_percentile": 5.0,   # Mais rigoroso
            "fps": 1,
        },
        "description": "Estilo fotorrealista com materiais PBR",
    },

    "low-poly": {
        "keywords": [
            "geometric",
            "minimalist",
            "flat shading",
            "low-poly mesh",
            "clean edges",
            "simple polygons",
            "faceted surfaces",
            "pastel colors",
        ],
        "negative_keywords": [
            "high-detail",
            "photorealistic",
            "organic",
            "complex textures",
        ],
        "params": {
            "target_size": 518,  # Múltiplo de 14 obrigatório (37 * 14 = 518)
            "edge_normal_threshold": 2.0,   # Arestas super definidas
            "edge_depth_threshold": 0.02,
            "confidence_percentile": 10.0,
            "fps": 1,
        },
        "description": "Estilo low-poly minimalista com formas geométricas limpas",
    },

    "sci-fi": {
        "keywords": [
            "futuristic",
            "sci-fi architecture",
            "neon lights",
            "cyberpunk",
            "metallic surfaces",
            "holographic elements",
            "advanced technology",
            "geometric patterns",
        ],
        "negative_keywords": [
            "medieval",
            "organic",
            "rustic",
            "natural",
        ],
        "params": {
            "target_size": 518,
            "edge_normal_threshold": 4.0,
            "edge_depth_threshold": 0.03,
            "confidence_percentile": 10.0,
            "fps": 1,
        },
        "description": "Estilo sci-fi futurista com elementos cyberpunk",
    },
}

# Mapeamento de aliases para nomes canônicos (aceita variações do usuário)
STYLE_ALIASES: dict = {
    # Minecraft
    "minecraft": "minecraft",
    "voxel": "minecraft",
    "blocky": "minecraft",
    "cubico": "minecraft",

    # RPG
    "rpg": "rpg",
    "fantasy": "rpg",
    "fantasia": "rpg",
    "medieval": "rpg",

    # Realistic
    "realistic": "realistic",
    "realista": "realistic",
    "realístico": "realistic",
    "real": "realistic",
    "photo": "realistic",

    # Low-Poly
    "low-poly": "low-poly",
    "lowpoly": "low-poly",
    "low_poly": "low-poly",
    "geometric": "low-poly",
    "geometrico": "low-poly",

    # Sci-Fi
    "sci-fi": "sci-fi",
    "scifi": "sci-fi",
    "futuristic": "sci-fi",
    "cyberpunk": "sci-fi",
    "futurista": "sci-fi",
}


class StyleManager:
    """
    Gerenciador de estilos para o pipeline World-to-Mesh.

    Converte prompts do usuário em prompts técnicos otimizados,
    injetando keywords e parâmetros que o HunyuanWorld entende melhor.

    Exemplo:
        >>> sm = StyleManager()
        >>> prompt, params = sm.enhance_prompt("uma vila medieval", "minecraft")
        >>> print(prompt)
        'A medieval village, voxel, blocky, 8-bit texture, cubic geometry, pixelated, ...'
    """

    def __init__(self, custom_presets: Optional[dict] = None):
        """
        Args:
            custom_presets: Presets customizados para sobrescrever/adicionar
                           estilos além dos padrões. Útil para experimentação.
        """
        self.presets = dict(STYLE_PRESETS)
        self.aliases = dict(STYLE_ALIASES)

        if custom_presets:
            self.presets.update(custom_presets)
            logger.info(f"Loaded {len(custom_presets)} custom style presets")

    # ── API Pública ────────────────────────────────────────────────────

    def enhance_prompt(
        self,
        user_prompt: str,
        style: str = "realistic",
    ) -> tuple:
        """
        Enriquece o prompt do usuário com keywords técnicas do estilo.

        A lógica interna:
        1. Resolve aliases (ex: "voxel" → "minecraft")
        2. Busca keywords do estilo
        3. Concatena ao prompt original
        4. Retorna parâmetros técnicos ajustados

        Args:
            user_prompt: Prompt original do usuário (qualquer idioma)
            style: Nome do estilo ou alias

        Returns:
            tuple: (enhanced_prompt: str, technical_params: dict)

        Raises:
            ValueError: Se o estilo não for reconhecido
        """
        # 1. Resolver alias para nome canônico
        canonical_style = self._resolve_style(style)

        preset = self.presets[canonical_style]
        keywords = preset["keywords"]
        params = preset["params"]

        # 2. Construir prompt enriquecido
        keyword_str = ", ".join(keywords)
        enhanced = f"{user_prompt}, {keyword_str}"

        logger.info(
            f"Style '{style}' → '{canonical_style}' | "
            f"Original: '{user_prompt}' | "
            f"Enhanced: '{enhanced}'"
        )

        return enhanced, dict(params)

    def get_negative_prompt(self, style: str) -> str:
        """
        Retorna prompt negativo para o estilo (termos a evitar).

        Args:
            style: Nome do estilo ou alias

        Returns:
            String com keywords negativas separadas por vírgula
        """
        canonical = self._resolve_style(style)
        negatives = self.presets[canonical].get("negative_keywords", [])
        return ", ".join(negatives)

    def list_styles(self) -> list:
        """Retorna lista de estilos disponíveis com descrições."""
        result = []
        for name, preset in self.presets.items():
            result.append({
                "name": name,
                "description": preset.get("description", ""),
                "keywords": preset["keywords"],
                "aliases": [
                    alias for alias, target in self.aliases.items()
                    if target == name and alias != name
                ],
            })
        return result

    def is_valid_style(self, style: str) -> bool:
        """Verifica se um estilo (ou alias) é válido."""
        return style.lower().strip() in self.aliases

    # ── Internos ───────────────────────────────────────────────────────

    def _resolve_style(self, style: str) -> str:
        """
        Resolve alias para nome canônico do estilo.

        Aceita variações de capitalização e espaços extras.
        """
        key = style.lower().strip()
        if key in self.aliases:
            return self.aliases[key]

        # Fallback: verificar se é nome canônico direto
        if key in self.presets:
            return key

        available = ", ".join(sorted(self.presets.keys()))
        raise ValueError(
            f"Estilo '{style}' não reconhecido. "
            f"Estilos disponíveis: {available}. "
            f"Aliases aceitos: {', '.join(sorted(self.aliases.keys()))}"
        )


# ── CLI Quick-Test ─────────────────────────────────────────────────────
# Permite testar o gerenciador rapidamente via terminal:
#   python -m src.utils.style_manager

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sm = StyleManager()

    print("=" * 60)
    print("  STYLE MANAGER – Teste Rápido")
    print("=" * 60)

    test_cases = [
        ("Uma vila medieval com castelo", "minecraft"),
        ("Floresta mágica com ruínas antigas", "rpg"),
        ("Montanha nevada com lago", "realistic"),
        ("Ilha tropical com palmeiras", "low-poly"),
        ("Base espacial orbital", "sci-fi"),
    ]

    for prompt, style in test_cases:
        enhanced, params = sm.enhance_prompt(prompt, style)
        negative = sm.get_negative_prompt(style)
        print(f"\n🎨 Estilo: {style}")
        print(f"   IN:  {prompt}")
        print(f"   OUT: {enhanced}")
        print(f"   NEG: {negative}")
        print(f"   Params: {params}")

    print("\n" + "=" * 60)
    print("  Estilos disponíveis:")
    for s in sm.list_styles():
        aliases = ", ".join(s["aliases"]) if s["aliases"] else "nenhum"
        print(f"  • {s['name']}: {s['description']} (aliases: {aliases})")
