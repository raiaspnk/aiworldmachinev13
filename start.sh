#!/bin/bash
# ==========================================================================
#  WORLD-TO-MESH UNIVERSAL - Auto Setup & Start
#  Este script instala dependências e baixa os pesos dos modelos.
# ==========================================================================

echo "🚀 Iniciando configuração do World-to-Mesh Universal..."

# 1. Instalação de dependências do Sistema (Linux/Ubuntu)
# Necessário para OpenCV, PyColmap e processamento de Mesh
echo "📦 Instalando dependências do sistema..."
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    cmake \
    build-essential \
    ninja-build \
    unzip \
    git-lfs \
    wget

git lfs install

# 2. Instalação de dependências Python
echo "🐍 Instalando bibliotecas Python (isso pode levar alguns minutos)..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install ninja

# 3. Download dos Pesos (Weights) - HunyunWorld-Mirror
echo "🌍 Baixando pesos do HunyuanWorld-Mirror..."
mkdir -p HunyuanWorld-Mirror-main/HunyuanWorld-Mirror-main/ckpts
huggingface-cli download tencent/HunyuanWorld-Mirror \
    --local-dir HunyuanWorld-Mirror-main/HunyuanWorld-Mirror-main/ckpts \
    --local-dir-use-symlinks False

# 4. Download dos Pesos (Weights) - Hunyuan3D-2
# O Hunyuan3D-2 costuma baixar via cache do HuggingFace (/.cache/huggingface)
# Mas vamos forçar o pré-download para não travar na primeira execução
echo "🔨 Baixando pesos do Hunyuan3D-2..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2')"

# 5. Compilar MonsterCore (C++/CUDA Engine)
echo "⚡ Compilando MonsterCore C++/CUDA..."
if python3 setup.py build_ext --inplace 2>&1; then
    echo "✅ MonsterCore compilado com sucesso!"
    python3 -c "import monster_core; monster_core.init_pool(1024); print('[MonsterCore] Teste de boot OK ✅')"
else
    echo "⚠️ MonsterCore falhou ao compilar (pipeline rodará em Python puro)"
fi

# 6. Permissões de Execução
chmod +x bridge.py

echo "================================================================"
echo "✅ CONFIGURAÇÃO CONCLUÍDA!"
echo "================================================================"
echo "Para gerar seu primeiro mundo 3D, use:"
echo "python3 bridge.py --prompt 'uma vila medieval' --style rpg"
echo "================================================================"

