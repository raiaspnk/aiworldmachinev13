#!/usr/bin/env python3
"""
==========================================================================
  TAG_DATASET.PY – Auto-tagger para Dataset de Geometria 3D
==========================================================================

Script para gerar arquivos .txt com tags técnicas para cada imagem do
dataset de treinamento World-to-Mesh.

Cada categoria (Cidades, Montanhas, Florestas, Scifi) recebe keywords
específicas focadas em reconstrução geométrica tridimensional.

Uso:
    python tag_dataset.py --dataset-dir ./meu_dataset
    python tag_dataset.py --dataset-dir ./dataset --dry-run

==========================================================================
"""

import argparse
from pathlib import Path
from typing import Dict, List

# ── Tags Base (todas as categorias) ────────────────────────────────────
BASE_TAGS = [
    "high fidelity geometry",
    "spatial reconstruction", 
    "depth-aware",
    "volumetric mesh",
    "3d scene reconstruction",
    "geometric precision",
]

# ── Tags Específicas por Categoria ────────────────────────────────────
CATEGORY_TAGS: Dict[str, List[str]] = {
    "Cidades": [
        "architectural wireframe",
        "urban topology",
        "building facade geometry",
        "street-level mesh",
        "structural edge detection",
        "man-made geometry",
    ],
    "Montanhas": [
        "heightmap relief",
        "terrain elevation mapping",
        "natural surface topology",
        "geological formation",
        "altitude gradient",
        "terrain mesh density",
    ],
    "Florestas": [
        "organic vegetation density",
        "foliage volume reconstruction",
        "natural canopy topology",
        "tree trunk geometry",
        "undergrowth layer depth",
        "biome spatial complexity",
    ],
    "Scifi": [
        "futuristic architecture",
        "technological geometry",
        "metallic surface reconstruction",
        "sci-fi structural topology",
        "advanced materials mesh",
        "otherworldly spatial design",
    ],
}

# Extensões de imagem suportadas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def generate_tags(category: str) -> str:
    """
    Gera string de tags para uma categoria específica.
    
    Args:
        category: Nome da categoria (Cidades, Montanhas, etc.)
    
    Returns:
        str: Tags separadas por vírgula
    """
    tags = BASE_TAGS.copy()
    
    if category in CATEGORY_TAGS:
        tags.extend(CATEGORY_TAGS[category])
    
    return ", ".join(tags)


def process_dataset(dataset_dir: Path, dry_run: bool = False) -> dict:
    """
    Processa o dataset gerando arquivos .txt para cada imagem.
    
    Args:
        dataset_dir: Diretório raiz do dataset
        dry_run: Se True, apenas simula sem criar arquivos
    
    Returns:
        dict: Estatísticas do processamento
    """
    stats = {
        "total_images": 0,
        "total_tags_created": 0,
        "by_category": {},
    }
    
    # Iterar sobre cada categoria
    for category_name in CATEGORY_TAGS.keys():
        category_dir = dataset_dir / category_name
        
        if not category_dir.exists():
            print(f"⚠️  Pasta não encontrada: {category_dir}")
            continue
        
        category_count = 0
        tags_text = generate_tags(category_name)
        
        # Processar todas as imagens da categoria
        for image_path in category_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            # Criar arquivo .txt com o mesmo nome da imagem
            txt_path = image_path.with_suffix(".txt")
            
            if dry_run:
                print(f"[DRY RUN] Criaria: {txt_path.name}")
            else:
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tags_text)
                    print(f"✅ {txt_path.name}")
                except Exception as e:
                    print(f"❌ Erro ao criar {txt_path.name}: {e}")
                    continue
            
            category_count += 1
            stats["total_tags_created"] += 1
        
        stats["total_images"] += category_count
        stats["by_category"][category_name] = category_count
        
        if category_count > 0:
            print(f"\n📁 {category_name}: {category_count} imagens processadas")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tagger para dataset World-to-Mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python tag_dataset.py --dataset-dir ./dataset\n"
            "  python tag_dataset.py --dataset-dir ./dataset --dry-run\n"
            "  python tag_dataset.py --dataset-dir ./dataset --show-tags Cidades\n"
        ),
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Diretório raiz do dataset (contendo as pastas de categorias)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo simulação: mostra o que seria criado sem criar arquivos",
    )
    parser.add_argument(
        "--show-tags",
        type=str,
        choices=list(CATEGORY_TAGS.keys()),
        help="Exibir as tags de uma categoria específica e sair",
    )
    
    args = parser.parse_args()
    
    # Modo exibição de tags
    if args.show_tags:
        print(f"\n📋 Tags para categoria '{args.show_tags}':\n")
        print(generate_tags(args.show_tags))
        return
    
    # Processar dataset
    dataset_path = Path(args.dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Diretório não encontrado: {dataset_path}")
        return
    
    print("=" * 60)
    print("  WORLD-TO-MESH DATASET AUTO-TAGGER")
    print("=" * 60)
    print(f"📂 Dataset: {dataset_path}")
    if args.dry_run:
        print("🔍 Modo: DRY RUN (simulação)")
    print("=" * 60)
    print()
    
    stats = process_dataset(dataset_path, dry_run=args.dry_run)
    
    # Exibir resumo
    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"Total de imagens: {stats['total_images']}")
    print(f"Total de tags criadas: {stats['total_tags_created']}")
    print("\nPor categoria:")
    for category, count in stats["by_category"].items():
        print(f"  • {category}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
