#!/bin/bash
# ==============================================================================
# AI WORLD ENGINE v14 + MONSTERCORE V2 - AUTO INSTALLER (RunPod Edition)
# Desenvolvido para RTX 4090 / L40S / A6000 (Ubuntu/Debian)
# ==============================================================================

set -e # Sai do script se qualquer comando falhar

echo "==================================================================="
echo "🚀 Iniciando Auto-Setup da AI World Engine (MonsterCore V2)..."
echo "==================================================================="

# 1. Atualizar SO e ferramentas básicas de compilação
echo "[1/6] 🛠️ Instalando pacotes de sistema e Ninja Build..."
sudo apt-get update -y
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 cmake build-essential ninja-build unzip git-lfs wget
git lfs install

# 2. Clonar os 12 Monstros (Diretórios de trabalho)
echo "[2/6] 📥 Clonando os Modelos de IA (Hunyuan3D, Real-ESRGAN, etc)..."
WORKSPACE_DIR="$(pwd)"
cd "$WORKSPACE_DIR"

if [ ! -d "Real-ESRGAN" ]; then git clone https://github.com/xinntao/Real-ESRGAN.git; fi
if [ ! -d "Hunyuan3D-2-main" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
    mv Hunyuan3D-2.1 Hunyuan3D-2-main
fi
if [ ! -d "StableNormal" ]; then git clone https://github.com/Stable-X/StableNormal.git; fi
if [ ! -d "HunyuanWorld-Mirror-main" ]; then
    git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git
    mv HunyuanWorld-Mirror HunyuanWorld-Mirror-main
fi

# 2.5 Injetar scripts customizados (World-to-Mesh) no repositório original clonado
if [ -d "custom_hw_scripts" ]; then
    echo "[2.5/6] 🧩 Injetando scripts customizados (StyleManager, etc) no HunyuanWorld..."
    cp custom_hw_scripts/infer.py HunyuanWorld-Mirror-main/infer.py
    cp -r custom_hw_scripts/src/* HunyuanWorld-Mirror-main/src/
fi

if [ ! -d "sd-scripts" ]; then git clone https://github.com/kohya-ss/sd-scripts.git; fi

# 3. Instalar PyTorch forçando CUDA 12.4+ (Para os Kernels do MonsterCore)
echo "[3/6] 🔥 Instalando PyTorch c/ CUDA 12.4 (Obrigatório para V2)..."
python3 -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# Instalar dependências adicionais
echo "[3.5/6] 📦 Instalando ecossistema de Visão e 3D Python..."
pip install -r requirements.txt || echo "requirements.txt pulado ou não encontrado"
pip install -r requirements-extra.txt || echo "requirements-extra.txt pulado ou não encontrado"
pip install ninja pyvista trimesh open3d pymeshlab basicsr timm transformers scipy xatlas huggingface_hub fastapi uvicorn

# 4. Compilar o MonsterCore V2 (C++/CUDA)
echo "[4/6] ⚙️ Compilando MonsterCore V2 (A Magia C++/CUDA)..."
if [ -f "setup.py" ] && [ -f "monster_core.cpp" ] && [ -f "monster_core_kernels.cu" ]; then
    chmod +x bridge.py setup.py
    python3 setup.py build_ext --inplace
    echo "✅ MonsterCore V2 Compilado!"
else
    echo "⚠️ AVISO: Scripts do MonsterCore (setup.py, .cpp, .cu) não encontrados. Envie-os para cá!"
fi

# 5. Baixar Pesos via HuggingFace
echo "[5/6] 🧠 Baixando Cérebro das IAs (Gigabytes de pesos)..."
mkdir -p "$WORKSPACE_DIR/weights" "$WORKSPACE_DIR/output" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/parts" "$WORKSPACE_DIR/sessions"

# Real-ESRGAN
if [ ! -f "$WORKSPACE_DIR/weights/RealESRGAN_x4plus.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P "$WORKSPACE_DIR/weights/"
fi

# Hunyuan3D e SAM 3
python3 -c "
import os
from huggingface_hub import snapshot_download, hf_hub_download
print('Iniciando HF Downloads...')
wk_dir = os.getcwd()
snapshot_download('tencent/Hunyuan3D-2', local_dir=f'{wk_dir}/weights/Hunyuan3D-2', resume_download=True)
"

# 6. Finalização e Teste do Motor
echo "[6/6] 🏁 Testando Ignição do MonsterCore..."
if python3 -c "import monster_core; monster_core.init_pool(8192); print('\n✅ MONSTERCORE DETECTADO NA PLACA DE VÍDEO!');" 2>/dev/null; then
    echo "==================================================================="
    echo "🏎️  AI WORLD ENGINE (V13) ESTÁ PRONTA!"
    echo "   Sua bateria de RTX 4090 está armada."
    echo ""
    echo "   Para gerar o primeiro asset digite:"
    echo "   python3 bridge.py --prompt 'castelo cyberpunk' --target-faces 150000 --smoothing-iterations 30"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "⚠️ SETUP CONCLUIDO, MAS O MONSTERCORE NÃO IMPORTOU CORRETAMENTE!"
    echo "   Verifique os logs de compilação acima."
    echo "==================================================================="
fi
