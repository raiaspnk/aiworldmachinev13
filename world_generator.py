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
    chunk_resolution: int = 1024,
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
        offset_x: Deslocamento no eixo X para Seamless Tiling de chunks adjacentes.
        offset_y: Deslocamento no eixo Y para Seamless Tiling de chunks adjacentes.
        scale: Escala física do World space (1.0 = bloco de 1x1 unidade).
    """
    print(f"🌍 [WorldGenerator] Iniciando extrusão de terreno ({chunk_resolution}x{chunk_resolution})...")
    
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
        
        # Extrair dimensões base da imagem RGB
        orig_w, orig_h = img_color.size
        # Validar tamanho da imagem vs chunk_resolution
        chunks_x = max(1, orig_w // chunk_resolution)
        chunks_y = max(1, orig_h // chunk_resolution)
        
        # Redimensionar para ficar exato múltiplo do chunk
        new_w = chunks_x * chunk_resolution
        new_h = chunks_y * chunk_resolution
        
        img_color = img_color.resize((new_w, new_h), Image.LANCZOS)
        img_depth_hpass = cv2.resize(img_depth_hpass, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"🌍 [WorldGenerator] Auto-Tiling ativado: {chunks_x}x{chunks_y} Grid. (Processing {chunks_x * chunks_y} chunks of {chunk_resolution}px)")
        
        # Iniciar container da Scene (para mesclar N mundos em 1 único arquivo exportado)
        scene = trimesh.Scene()
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Setup CUDA
        import torch
        import sys
        try:
            from monster_core import generate_world_geometry_pipeline
            use_cuda = torch.cuda.is_available()
        except ImportError:
            print("⚠️ [WorldGenerator] Aviso: monster_core V3 não encontrado. Fallback inviável para multi-tiling.")
            use_cuda = False
            
        if not use_cuda:
            print("❌ Erro: Gerar mundos massivos requer MonsterCore V3 CUDA ativo.")
            return False
            
        # Loop de Tiling (Gerar Fatias)
        for cy in range(chunks_y):
            for cx in range(chunks_x):
                print(f"   => Forging Chunk [{cx}, {cy}]...")
                
                # Coordenadas Crop
                x0 = cx * chunk_resolution
                y0 = cy * chunk_resolution
                x1 = x0 + chunk_resolution
                y1 = y0 + chunk_resolution
                
                # 1.1 Fatiar Textura e Profundidade
                crop_color = img_color.crop((x0, y0, x1, y1))
                crop_depth = img_depth_hpass[y0:y1, x0:x1]
                
                # Normalização adaptativa por chunk
                max_val = np.max(crop_depth)
                if max_val == 0: max_val = 1.0
                depth_norm = crop_depth.astype(np.float32) / float(max_val)
                depth_tensor = torch.from_numpy(depth_norm).cuda()
                
                # 2. Gerar Geometria V3 (Zero Overhead)
                # Aplicamos offset_x e offset_y físicos para as fatias se conectarem 1:1 no espaço 3D
                offset_x_world = float(cx) * 1.0
                offset_y_world = float(chunks_y - 1 - cy) * 1.0 # Inverte eixo Y para bater c/ a imagem
                
                vertices_tensor, faces_tensor, normals_tensor, foliage_tensor = generate_world_geometry_pipeline(
                    depth_map=depth_tensor, 
                    max_height=max_height,
                    offset_x=offset_x_world,
                    offset_y=offset_y_world,
                    scale=1.0,
                    smooth_iters=smoothing_iterations,
                    smooth_lambda=0.5
                )
                
                vertices = vertices_tensor.cpu().numpy()
                faces = faces_tensor.cpu().numpy()
                normals = normals_tensor.cpu().numpy()
                foliage_mask = foliage_tensor.cpu().numpy()
                
                # 3. Coordenadas UV e Malha parciais
                u = np.linspace(0, 1, chunk_resolution)
                v = np.linspace(1, 0, chunk_resolution)
                uu, vv = np.meshgrid(u, v)
                uvs = np.column_stack((uu.flatten(), vv.flatten()))
                
                mesh_chunk = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # Salvar Normal Map parciais associados ao chunk
                normal_path = os.path.join(base_dir, f"{base_name}_normal_{cx}_{cy}.png")
                normals_img = (normals * 255).astype(np.uint8)
                cv2.imwrite(normal_path, cv2.cvtColor(normals_img, cv2.COLOR_RGB2BGR))
                img_normal = Image.open(normal_path).convert("RGB")
                
                # Salvar Foliage Mask parciais associados ao chunk
                foliage_path = os.path.join(base_dir, f"{base_name}_foliage_{cx}_{cy}.png")
                foliage_img = (foliage_mask * 255).astype(np.uint8)
                kernel_morph = np.ones((5,5), np.uint8)
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_OPEN, kernel_morph) 
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_CLOSE, kernel_morph)
                cv2.imwrite(foliage_path, foliage_img)
                
                # Construir material PBR real (Principled BSDF) para o chunk
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=crop_color,
                    normalTexture=img_normal,
                    roughnessFactor=0.8,
                    metallicFactor=0.1
                )
                
                mesh_chunk.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
                # Adicionar chunk a grande cena exportável
                scene.add_geometry(mesh_chunk, geom_name=f"chunk_{cx}_{cy}")
                
        # 5. Exportar GLB massivo único
        print(f"🌍 [WorldGenerator] Exportando mundo inteiro conectado (Scene) para: {output_path}")
        scene.export(output_path)
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
    parser = argparse.ArgumentParser(description="AI World Engine - V3 Generator")
    parser.add_argument("--image", required=True, help="Imagem RGB base (Texture)")
    parser.add_argument("--depth", required=True, help="Mapa de Profundidade")
    parser.add_argument("--output", required=True, help="Saída GLB")
    parser.add_argument("--chunk_resolution", type=int, default=1024, help="Resolução de cada lote do grid de VRAM (Ex: 1024)")
    args = parser.parse_args()
    
    generate_landscape_mesh(
        args.image, args.depth, args.output, chunk_resolution=args.chunk_resolution
    )
