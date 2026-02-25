import os
import cv2
import trimesh
import numpy as np
from PIL import Image

def generate_landscape_mesh(
    image_path: str,
    depth_map_path: str,
    output_path: str,
    max_height: float = 0.5,
    mesh_resolution: int = 1024,
    smoothing_iterations: int = 3
) -> bool:
    """
    Gera uma malha 3D (.glb) de um cenário completo aplicando "Depth Displacement".
    Pega uma grade 2D plana, e empurra os vértices no eixo Z baseado no tom de cinza
    do mapa de profundidade, aplicando a textura colorida original por cima.
    
    Args:
        image_path: Caminho da imagem RGB original (textura).
        depth_map_path: Caminho do mapa de profundidade (grayscale).
        output_path: Caminho final do arquivo .glb.
        max_height: Altura máxima de extrusão (baseado no depth 255).
        mesh_resolution: Quantidade X e Y de polígonos da malha (ex: 512x512).
        smoothing_iterations: Iterações de Laplacian Smoothing para não ficar pontiagudo.
    """
    print(f"🌍 [WorldGenerator] Iniciando extrusão de terreno ({mesh_resolution}x{mesh_resolution})...")
    
    try:
        if not os.path.exists(depth_map_path):
            print(f"❌ [WorldGenerator] Erro: Mapa de profundidade não encontrado: {depth_map_path}")
            return False
            
        # 1. Carregar a textura e o depth map
        img_color = Image.open(image_path).convert("RGB")
        
        # [ANTI-STAIRCASE] Ler em 16-bits para evitar quantização e achatamento (Efeito Escada)
        img_depth = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
        
        # [ANTI-MELTING] Aplicar filtro High-Pass (Sharpen) na profundidade
        # Isso acentua as quinas dos prédios e "cliva" transições suaves entre parede e chão
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_depth_hpass = cv2.filter2D(img_depth, -1, kernel_sharpen)
        
        # Redimensionar depth map para a nova alta resolução 1024x1024 (1M faces) usando Lanczos4 para preservar bordas finas
        img_depth_resized = cv2.resize(img_depth_hpass, (mesh_resolution, mesh_resolution), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalização adaptativa (Suporta 8-bit ou 16-bit)
        max_val = np.max(img_depth_resized)
        if max_val == 0: max_val = 1.0 # Evitar divisão por zero
        img_depth_normalized = img_depth_resized.astype(np.float32) / float(max_val)
        
        # Inverter o mapa se necessário (depende se o DepthAnything pretoo=longe ou branco=longe)
        # Depth Anything V2: Branco = perto, Preto = longe.
        # Nós queremos que o mais próximo suba no Z.
        
        # 2. Criar a Grade Plana e Extrudar (VRAM PTX CUDA)
        print("🌍 [WorldGenerator] Instancionado Terreno Nível PTX/CUDA (MonsterCore)...")
        import torch
        import sys
        
        try:
            from monster_core import generate_displaced_grid
            use_cuda = torch.cuda.is_available()
        except ImportError:
            print("⚠️ [WorldGenerator] Aviso: monster_core não encontado. Fallback para Numpy.")
            use_cuda = False
            
        if use_cuda:
            # 2.1 Enviar profundidade para VRAM
            depth_tensor = torch.from_numpy(img_depth_normalized).cuda()
            
            # 2.2 Gerar matrizes de geometria na GPU via C++
            # V2: O NVidia Kernel agora calcula Vetores Normais em paralalelo
            vertices_tensor, faces_tensor, normals_tensor = generate_displaced_grid(depth_tensor, max_height)
            
            # 2.3 [ANTI-MELTING] Suavização Bilateral GPU
            # O kernel C++ executa o Laplacian com Threshold Bilateral (Ignora as paredes 90 graus, suaviza apenas o chão/terreno organicamente)
            if smoothing_iterations > 0:
                print(f"🌍 [WorldGenerator] Suavização Bilateral PTX ativa (Iterations: {smoothing_iterations})...")
                # MonsterCore.laplacian_smooth_csr_kernel (Threshold hardcoded no .cu previne derretimento)
                from monster_core import laplacian_smooth
                # Suavizar apenas posições (vetores). Faces são re-passadas intactas.
                vertices_tensor = laplacian_smooth(vertices_tensor, faces_tensor, iterations=smoothing_iterations, lambda_factor=0.5)
            
            # 2.4 Resgatar para RAM em C++ Pinned Memory
            vertices = vertices_tensor.cpu().numpy()
            faces = faces_tensor.cpu().numpy()
            normals = normals_tensor.cpu().numpy() # [H, W, 3] Image format
            
            print(f"🌍 [WorldGenerator] Memória do terreno: {len(vertices)} vértices | {len(faces)} polígonos.")
            
        else:
            # Fallback (Antigo CPU)
            x = np.linspace(-0.5, 0.5, mesh_resolution)
            y = np.linspace(-0.5, 0.5, mesh_resolution)
            xx, yy = np.meshgrid(x, y)
            zz = img_depth_normalized * max_height
            vertices = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
            
            idx = np.arange(mesh_resolution * mesh_resolution).reshape(mesh_resolution, mesh_resolution)
            t1 = np.column_stack((idx[:-1, :-1].flatten(), idx[1:, :-1].flatten(), idx[:-1, 1:].flatten()))
            t2 = np.column_stack((idx[1:, :-1].flatten(), idx[1:, 1:].flatten(), idx[:-1, 1:].flatten()))
            faces = np.vstack((t1, t2))
        
        # Coordenadas UV (0 a 1) para mapear a textura (sempre na CPU pq trimesh não liga)
        u = np.linspace(0, 1, mesh_resolution)
        v = np.linspace(1, 0, mesh_resolution) # Invertido no Y para alinhar com imagem
        uu, vv = np.meshgrid(u, v)
        uvs = np.column_stack((uu.flatten(), vv.flatten()))
        
        # 3. Construir o objeto trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 3.1 PBR Materials (The Great Forging)
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        if use_cuda:
            # Salvar Normal Map gerado pela VRAM
            normal_path = os.path.join(base_dir, f"{base_name}_normal.png")
            # Converter BGR para RGB (Padrão OpenGL) e 255
            normals_img = (normals * 255).astype(np.uint8)
            cv2.imwrite(normal_path, cv2.cvtColor(normals_img, cv2.COLOR_RGB2BGR))
            img_normal = Image.open(normal_path).convert("RGB")
            
            # Construir material PBR real (Principled BSDF)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=img_color,
                normalTexture=img_normal,
                roughnessFactor=0.8,
                metallicFactor=0.1
            )
            print(f"🎨 [WorldGenerator] Embedded PBR Normal Map: {normal_path}")
        else:
            material = trimesh.visual.material.SimpleMaterial(image=img_color)
            
        # Configurar coordenas UV no visual
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        
        # 4. Suavização (GPU Laplacian via MonsterCore já ocorre internamente pelo pipeline se invocado depois)
        # O trimesh.smoothing faria tudo derreter, portanto removemos as iterações aqui para preservar o threshold Bilateral do PTX.
        # Caso precise, isso deve ser roteado pelo async_geometry_pipeline que acabamos de alterar.
        if smoothing_iterations > 0 and not use_cuda:
            print(f"🌍 [WorldGenerator] (Fallback) Suavizando terreno via CPU...")
            trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)
            
        # 5. Exportar GLB
        print(f"🌍 [WorldGenerator] Exportando para: {output_path}")
        mesh.export(output_path)
        print("✅ [WorldGenerator] Terreno gerado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ [WorldGenerator] Falha crítica ao gerar landscape: {e}")
        import traceback
        traceback.print_exc()
        return False

# Para testes isolados
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_landscape_mesh(args.image, args.depth, args.output)
