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
    smoothing_iterations: int = 3,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0
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
        
        # [ANTI-STAIRCASE] Ler NPZ bruto métrico (float) ou Imagem 16-bits
        if depth_map_path.endswith('.npz'):
            depth_data = np.load(depth_map_path)
            img_depth = depth_data['depth']
            
            # INVERTER PROFUNDIDADE MÉTRICA: Elementos próximos (valores menores)
            # devem se tornar as extrusões mais altas do terreno (valores maiores).
            min_d, max_d = img_depth.min(), img_depth.max()
            if max_d > min_d:
                img_depth = (max_d - img_depth) / (max_d - min_d)
            else:
                img_depth = np.zeros_like(img_depth)
        else:
            img_depth = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
            
        if img_depth is None:
            print(f"❌ [WorldGenerator] Erro: Falha ao ler mapa de profundidade: {depth_map_path}")
            return False
            
        # Padronizar como float32 para o filtro matricial
        img_depth = img_depth.astype(np.float32)
        
        # [ANTI-MELTING] Aplicar filtro High-Pass (Sharpen) na profundidade
        # Isso acentua as quinas dos prédios e "cliva" transições suaves entre parede e chão
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_depth_hpass = cv2.filter2D(img_depth, -1, kernel_sharpen)
        
        # Extrair dimensões base da imagem RGB
        orig_w, orig_h = img_color.size
        
        # Para garantir que os tiles não tenham UMA fresta de divisão, precisamos que o último 
        # vértice do Tile 0 seja IDÊNTICO na posição X,Y,Z ao primeiro vértice do Tile 1.
        # Portanto, há um Overlap obrigatório de 1 pixel em toda as junções.
        overlap = 1
        eff_res = chunk_resolution - overlap
        
        # Validar tamanho da imagem vs chunk_resolution
        chunks_x = max(1, orig_w // eff_res)
        chunks_y = max(1, orig_h // eff_res)
        
        # Redimensionar para o novo tamanho costurado
        new_w = chunks_x * eff_res + overlap
        new_h = chunks_y * eff_res + overlap
        
        img_color = img_color.resize((new_w, new_h), Image.LANCZOS)
        img_depth_hpass = cv2.resize(img_depth_hpass, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"🌍 [WorldGenerator] Seamless Auto-Tiling ativado: {chunks_x}x{chunks_y} Grid. (Processing {chunks_x * chunks_y} chunks of {chunk_resolution}px with 1px structural overlap)")
        
        # Iniciar container da Scene (para mesclar N mundos em 1 único arquivo exportado)
        scene = trimesh.Scene()
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Setup CUDA
        import torch
        import torch.nn.functional as F
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
                
                # Coordenadas Crop com Overlap Matemático (evita rasgar a malha no horizonte)
                x0 = cx * eff_res
                y0 = cy * eff_res
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
                
                # --- [MONSTER CORE V4: Semantic Geometry Decoupling] ---
                import torch.nn.functional as F
                kernel_sz = 31
                pad = kernel_sz // 2
                depth_unsqueezed = depth_tensor.unsqueeze(0).unsqueeze(0)
                eroded = -F.max_pool2d(-depth_unsqueezed, kernel_sz, stride=1, padding=pad)
                terrain_depth_tensor = F.max_pool2d(eroded, kernel_sz, stride=1, padding=pad).squeeze()
                
                # Mascara Booleana: Prédios/Estruturas (Altura dif > 2% do chão)
                building_mask_tensor = (depth_tensor - terrain_depth_tensor) > 0.02
                
                overlap_compensation = float(eff_res) / float(chunk_resolution - 1)
                offset_x_world = float(cx) * overlap_compensation * scale
                offset_y_world = float(chunks_y - 1 - cy) * overlap_compensation * scale # Inverte eixo Y
                
                # Ajustar max_height pela escala para manter a proporção (se scale=100, altura máxima=50)
                actual_max_height = max_height * scale
                
                # 2.A: Forjar Geometria do Chão Limpo (Terrain Mesh)
                t_verts_pt, t_faces_pt, t_norms_pt, t_foliage_pt, _ = generate_world_geometry_pipeline(
                    depth_map=terrain_depth_tensor, 
                    max_height=actual_max_height, offset_x=offset_x_world, offset_y=offset_y_world,
                    scale=scale, smooth_iters=smoothing_iterations, smooth_lambda=0.5
                )
                
                # 2.B: Forjar Geometria da Cidade (Prédios e Muros Triplanares)
                b_verts_pt, b_faces_pt, b_norms_pt, b_foliage_pt, b_mat_pt = generate_world_geometry_pipeline(
                    depth_map=depth_tensor, 
                    max_height=actual_max_height, offset_x=offset_x_world, offset_y=offset_y_world,
                    scale=scale, smooth_iters=0, smooth_lambda=0.5 # Muros retos, não suavizar prédios
                )
                
                t_verts = t_verts_pt.cpu().numpy()
                t_faces = t_faces_pt.cpu().numpy()
                
                b_verts = b_verts_pt.cpu().numpy()
                b_faces = b_faces_pt.cpu().numpy()
                b_mat = b_mat_pt.cpu().numpy()
                bldg_mask = building_mask_tensor.cpu().numpy().flatten()
                
                # --- Filtro V4: Isolar Prédios e Muros ---
                face_v0 = b_faces[:, 0]
                face_v1 = b_faces[:, 1]
                face_v2 = b_faces[:, 2]
                
                is_roof = bldg_mask[face_v0] | bldg_mask[face_v1] | bldg_mask[face_v2]
                is_wall = (b_mat == 1)
                
                bldg_roof_faces = b_faces[is_roof & ~is_wall]
                bldg_wall_faces = b_faces[is_wall]
                
                # 3. Coordenadas UV (Mesmo mapeamento Top-Down)
                u = np.linspace(0, 1, chunk_resolution)
                v = np.linspace(1, 0, chunk_resolution)
                uu, vv = np.meshgrid(u, v)
                uvs = np.column_stack((uu.flatten(), vv.flatten()))
                
                normal_path = os.path.join(base_dir, f"{base_name}_normal_{cx}_{cy}.png")
                normals_img = (t_norms_pt.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(normal_path, cv2.cvtColor(normals_img, cv2.COLOR_RGB2BGR))
                img_normal = Image.open(normal_path).convert("RGB")
                
                foliage_path = os.path.join(base_dir, f"{base_name}_foliage_{cx}_{cy}.png")
                foliage_img = (t_foliage_pt.cpu().numpy() * 255).astype(np.uint8)
                kernel_morph = np.ones((5,5), np.uint8)
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_OPEN, kernel_morph) 
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_CLOSE, kernel_morph)
                cv2.imwrite(foliage_path, foliage_img)
                
                # Material Base Top-Down
                material_td = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=crop_color,
                    normalTexture=img_normal,
                    roughnessFactor=0.8,
                    metallicFactor=0.1
                )
                
                # Material Triplanar (Para os Muros Cortados da Cidade)
                material_wall = trimesh.visual.material.PBRMaterial(
                    name=f"Triplanar_Wall",
                    baseColorFactor=[100, 100, 100, 255],
                    metallicFactor=0.2,
                    roughnessFactor=0.5
                )

                # Construir Meshes
                t_mesh = trimesh.Trimesh(vertices=t_verts, faces=t_faces, process=False)
                t_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material_td)
                scene.add_geometry(t_mesh, geom_name=f"Terrain_{cx}_{cy}")
                
                if len(bldg_roof_faces) > 0:
                    b_roof = trimesh.Trimesh(vertices=b_verts, faces=bldg_roof_faces, process=False)
                    b_roof.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material_td)
                    scene.add_geometry(b_roof, geom_name=f"BldgRoof_{cx}_{cy}")
                    
                if len(bldg_wall_faces) > 0:
                    b_wall = trimesh.Trimesh(vertices=b_verts, faces=bldg_wall_faces, process=False)
                    b_wall.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material_wall)
                    scene.add_geometry(b_wall, geom_name=f"BldgWall_{cx}_{cy}")
                
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
