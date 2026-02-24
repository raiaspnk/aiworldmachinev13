// ============================================================================
//  MONSTER_KERNELS.CU – Pure CUDA Kernels para o MonsterCore v2
// ============================================================================
//
//  Kernels nativos que rodam diretamente nos CUDA cores, sem overhead
//  do PyTorch tensor API. Performance máxima para operações geométricas.
//
//  Componentes:
//    1. ransac_distance_kernel    – Distância ponto-plano massivamente paralela
//    2. laplacian_smooth_kernel   – Smooth por vértice em um único launch
//    3. quadric_error_kernel      – Cálculo de erro quadrático para decimation
//    4. pinned_transfer_kernel    – Helper para memória pinned
//
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <iostream>
#include <cmath>

// ============================================================================
// CONSTANTES E HELPERS
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Macro para checar erros CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[MonsterCore CUDA] Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// KERNEL 1: RANSAC Distance – Distância Ponto-Plano Paralela
// ============================================================================
//
// Cada thread calcula a distância de UM vértice a UM plano candidato.
// Grid: [num_candidates, ceil(num_points / BLOCK_SIZE)]
// Resultado: matriz [num_candidates x num_points] de distâncias
//
// ============================================================================

__global__ void ransac_distance_kernel(
    const float* __restrict__ vertices,   // [N, 3]
    const float* __restrict__ plane_points, // [C, 3] pontos base de cada plano
    const float* __restrict__ plane_normals, // [C, 3] normais de cada plano
    float* __restrict__ distances,        // [C, N] distâncias
    int* __restrict__ inlier_counts,      // [C] contadores atômicos
    const float threshold,
    const int num_points,
    const int num_candidates
) {
    int candidate_idx = blockIdx.x;
    int point_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (candidate_idx >= num_candidates || point_idx >= num_points) return;

    // Carregar normal e ponto base do plano candidato
    float nx = plane_normals[candidate_idx * 3 + 0];
    float ny = plane_normals[candidate_idx * 3 + 1];
    float nz = plane_normals[candidate_idx * 3 + 2];
    float px = plane_points[candidate_idx * 3 + 0];
    float py = plane_points[candidate_idx * 3 + 1];
    float pz = plane_points[candidate_idx * 3 + 2];

    // Carregar vértice
    float vx = vertices[point_idx * 3 + 0];
    float vy = vertices[point_idx * 3 + 1];
    float vz = vertices[point_idx * 3 + 2];

    // Distância ponto-plano: |dot(v - p, n)|
    float dx = vx - px;
    float dy = vy - py;
    float dz = vz - pz;
    float dist = fabsf(dx * nx + dy * ny + dz * nz);

    // Escrever distância
    distances[candidate_idx * num_points + point_idx] = dist;

    // Contar inliers com atomicAdd (só conta se abaixo do threshold)
    if (dist < threshold) {
        atomicAdd(&inlier_counts[candidate_idx], 1);
    }
}

// ============================================================================
// KERNEL 2: Plane Projection – Projetar inliers no plano
// ============================================================================

__global__ void plane_projection_kernel(
    float* __restrict__ vertices,        // [N, 3] (in-place)
    const float* __restrict__ distances, // [N] distâncias ao melhor plano
    const float nx, const float ny, const float nz, // Normal do plano
    const float threshold,
    const int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float dist = distances[idx];
    if (dist < threshold) {
        // Projetar vértice no plano: v -= dist * normal
        // Calcular a distância com sinal para a projeção correta
        float vx = vertices[idx * 3 + 0];
        float vy = vertices[idx * 3 + 1];
        float vz = vertices[idx * 3 + 2];

        // Recalcular distância com sinal (não absoluta)
        // d_signed = dot(v - p, n) — mas aqui usamos o fato de que
        // dist = |d_signed|, e o sinal é determinado pela direção
        // Simplificação: usamos o distances array que já tem os valores
        vertices[idx * 3 + 0] -= dist * nx;
        vertices[idx * 3 + 1] -= dist * ny;
        vertices[idx * 3 + 2] -= dist * nz;
    }
}

// ============================================================================
// KERNEL 3: Laplacian Smooth – Um vértice por thread
// ============================================================================
//
// Cada thread processa um vértice:
//   1. Percorre os vizinhos (via edge list)
//   2. Calcula a média das posições
//   3. Interpola: v_new = (1-λ)v + λ·mean(vizinhos)
//
// ============================================================================

__global__ void laplacian_smooth_kernel(
    const float* __restrict__ vertices_in,  // [N, 3]
    float* __restrict__ vertices_out,       // [N, 3]
    const int* __restrict__ adj_offsets,     // [N+1] CSR offsets
    const int* __restrict__ adj_indices,     // [E] vizinhos
    const float lambda_factor,
    const int num_verts
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_verts) return;

    int start = adj_offsets[vid];
    int end = adj_offsets[vid + 1];
    int degree = end - start;

    float vx = vertices_in[vid * 3 + 0];
    float vy = vertices_in[vid * 3 + 1];
    float vz = vertices_in[vid * 3 + 2];

    if (degree == 0) {
        // Vértice isolado — mantém posição
        vertices_out[vid * 3 + 0] = vx;
        vertices_out[vid * 3 + 1] = vy;
        vertices_out[vid * 3 + 2] = vz;
        return;
    }

    // Acumular posições dos vizinhos
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    for (int i = start; i < end; ++i) {
        int neighbor = adj_indices[i];
        sum_x += vertices_in[neighbor * 3 + 0];
        sum_y += vertices_in[neighbor * 3 + 1];
        sum_z += vertices_in[neighbor * 3 + 2];
    }

    // Média dos vizinhos
    float inv_degree = 1.0f / static_cast<float>(degree);
    float mean_x = sum_x * inv_degree;
    float mean_y = sum_y * inv_degree;
    float mean_z = sum_z * inv_degree;

    // Interpolação: v_new = (1-λ)v + λ·mean
    vertices_out[vid * 3 + 0] = (1.0f - lambda_factor) * vx + lambda_factor * mean_x;
    vertices_out[vid * 3 + 1] = (1.0f - lambda_factor) * vy + lambda_factor * mean_y;
    vertices_out[vid * 3 + 2] = (1.0f - lambda_factor) * vz + lambda_factor * mean_z;
}

// ============================================================================
// KERNEL 4: Quadric Error Metric – Para Edge Collapse (Decimation)
// ============================================================================
//
// Para cada aresta (v0, v1), calcula o erro quadrático Q de Garland-Heckbert.
// O erro Q mede o custo de colapsar essa aresta em um único vértice.
// Arestas com erro menor são colapsadas primeiro.
//
// ============================================================================

__global__ void compute_face_quadrics_kernel(
    const float* __restrict__ vertices,     // [N, 3]
    const int* __restrict__ faces,          // [F, 3]
    float* __restrict__ quadrics,           // [N, 10] (symmetric 4x4 matrix compactada)
    const int num_faces
) {
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    // Indices dos vértices da face
    int i0 = faces[fid * 3 + 0];
    int i1 = faces[fid * 3 + 1];
    int i2 = faces[fid * 3 + 2];

    // Posições
    float v0x = vertices[i0 * 3 + 0], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
    float v1x = vertices[i1 * 3 + 0], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
    float v2x = vertices[i2 * 3 + 0], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];

    // Edges
    float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // Normal do triângulo via cross product
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;

    // Normalizar
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len < 1e-10f) return;
    float inv_len = 1.0f / len;
    nx *= inv_len; ny *= inv_len; nz *= inv_len;

    // Equação do plano: ax + by + cz + d = 0
    float d = -(nx * v0x + ny * v0y + nz * v0z);

    // Computar Q = p * p^T (simétrica 4x4, armazenada como 10 floats)
    // Q = [a²  ab  ac  ad]
    //     [ab  b²  bc  bd]
    //     [ac  bc  c²  cd]
    //     [ad  bd  cd  d²]
    float q[10] = {
        nx*nx, nx*ny, nx*nz, nx*d,
               ny*ny, ny*nz, ny*d,
                      nz*nz, nz*d,
                             d*d
    };

    // Acumular Q em cada vértice da face via atomicAdd
    int vert_ids[3] = {i0, i1, i2};
    for (int v = 0; v < 3; ++v) {
        int vid = vert_ids[v];
        for (int qi = 0; qi < 10; ++qi) {
            atomicAdd(&quadrics[vid * 10 + qi], q[qi]);
        }
    }
}

// ============================================================================
// KERNEL 5: Edge Collapse Cost – Avalia custo de cada aresta
// ============================================================================

__global__ void compute_edge_collapse_cost_kernel(
    const float* __restrict__ vertices,     // [N, 3]
    const float* __restrict__ quadrics,     // [N, 10]
    const int* __restrict__ edges,          // [E, 2]
    float* __restrict__ costs,              // [E]
    float* __restrict__ optimal_pos,        // [E, 3] posição ótima
    const int num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;

    int v0 = edges[eid * 2 + 0];
    int v1 = edges[eid * 2 + 1];

    // Q_sum = Q_v0 + Q_v1
    float Q[10];
    for (int i = 0; i < 10; ++i) {
        Q[i] = quadrics[v0 * 10 + i] + quadrics[v1 * 10 + i];
    }

    // Para simplicidade, usamos o ponto médio como posição ótima
    // (resolver o sistema linear 4x4 no kernel seria possível mas complexo)
    float mx = (vertices[v0 * 3 + 0] + vertices[v1 * 3 + 0]) * 0.5f;
    float my = (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1]) * 0.5f;
    float mz = (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2]) * 0.5f;

    // Calcular erro quadrático: v^T Q v
    // Q é simétrica 4x4 representada como 10 valores:
    // [q0  q1  q2  q3]     indices: 0 1 2 3
    // [q1  q4  q5  q6]              1 4 5 6
    // [q2  q5  q7  q8]              2 5 7 8
    // [q3  q6  q8  q9]              3 6 8 9
    float cost = Q[0]*mx*mx + 2*Q[1]*mx*my + 2*Q[2]*mx*mz + 2*Q[3]*mx
               + Q[4]*my*my + 2*Q[5]*my*mz + 2*Q[6]*my
               + Q[7]*mz*mz + 2*Q[8]*mz
               + Q[9];

    costs[eid] = cost;
    optimal_pos[eid * 3 + 0] = mx;
    optimal_pos[eid * 3 + 1] = my;
    optimal_pos[eid * 3 + 2] = mz;
}

// ============================================================================
// HOST WRAPPERS – Funções C++ que lançam os kernels
// ============================================================================

// ------ RANSAC MULTI-PLANO COM CUDA STREAMS ------

torch::Tensor cuda_ransac_multiplane(
    torch::Tensor vertices,
    float distance_threshold,
    int num_iterations,
    int batch_size,
    int max_planes
) {
    TORCH_CHECK(vertices.is_cuda(), "Vértices devem estar na GPU.");
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "Shape: [N, 3]");

    auto verts = vertices.to(torch::kFloat32).contiguous();
    int num_points = verts.size(0);
    auto result = verts.clone();

    int planes_found = 0;
    auto active_mask = torch::ones({num_points}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // Criar CUDA stream dedicado
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int plane = 0; plane < max_planes; ++plane) {
        // Pegar apenas os vértices ativos (não retificados ainda)
        auto active_indices = torch::nonzero(active_mask).squeeze(1);
        int num_active = active_indices.size(0);
        if (num_active < 100) break;

        auto active_verts = result.index_select(0, active_indices).contiguous();

        int64_t global_best_count = 0;
        int global_best_batch_idx = -1;
        torch::Tensor best_inlier_counts;
        torch::Tensor best_distances;
        torch::Tensor best_normals;
        torch::Tensor best_points;

        // Mini-batching RANSAC
        for (int b = 0; b < num_iterations; b += batch_size) {
            int current_batch = std::min(batch_size, num_iterations - b);

            // Sortear 3 pontos
            auto indices = torch::randint(0, num_active, {current_batch, 3},
                torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
            auto idx_flat = indices.flatten();
            auto pts = active_verts.index_select(0, idx_flat).view({current_batch, 3, 3});

            auto p1 = pts.select(1, 0);
            auto p2 = pts.select(1, 1);
            auto p3 = pts.select(1, 2);

            auto v1 = p2 - p1;
            auto v2 = p3 - p1;
            auto normals = torch::cross(v1, v2, 1);
            auto norms = normals.norm(2, 1, true);
            auto valid = norms.squeeze(1) > 1e-8;
            normals = normals / (norms + 1e-10);

            // Preparar dados para o kernel CUDA
            auto plane_points_contig = p1.contiguous();
            auto plane_normals_contig = normals.contiguous();

            // Alocar output
            auto dist_matrix = torch::zeros({current_batch, num_active},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto counts = torch::zeros({current_batch},
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

            // Lançar kernel de distância
            dim3 grid(current_batch, (num_active + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block(BLOCK_SIZE);

            ransac_distance_kernel<<<grid, block, 0, stream>>>(
                active_verts.data_ptr<float>(),
                plane_points_contig.data_ptr<float>(),
                plane_normals_contig.data_ptr<float>(),
                dist_matrix.data_ptr<float>(),
                counts.data_ptr<int>(),
                distance_threshold,
                num_active,
                current_batch
            );
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Invalidar planos degenerados
            counts.masked_fill_(~valid, 0);

            // Melhor desta batch
            int best_idx = torch::argmax(counts).item<int>();
            int64_t best_count = counts[best_idx].item<int64_t>();

            if (best_count > global_best_count) {
                global_best_count = best_count;
                global_best_batch_idx = best_idx;
                best_distances = dist_matrix[best_idx].clone();
                best_normals = normals[best_idx].clone();
                best_points = p1[best_idx].clone();
            }
        }

        if (global_best_count < static_cast<int64_t>(num_active * 0.05)) break;

        // Projetar inliers no plano via kernel
        auto inlier_mask = best_distances < distance_threshold;
        auto inlier_indices_local = torch::nonzero(inlier_mask).squeeze(1);
        auto global_inlier_indices = active_indices.index_select(0, inlier_indices_local);

        // Usar kernel de projeção
        auto inlier_verts = result.index_select(0, global_inlier_indices);
        auto diff = inlier_verts - best_points.unsqueeze(0);
        auto signed_dist = torch::sum(diff * best_normals.unsqueeze(0), 1, true);
        auto projected = inlier_verts - signed_dist * best_normals.unsqueeze(0);

        result.index_copy_(0, global_inlier_indices, projected);

        // Mascarar vértices já retificados
        active_mask.index_fill_(0, global_inlier_indices, false);
        planes_found++;

        std::cout << "[MonsterCore v2] Plano " << planes_found
                  << ": " << global_best_count << " vértices retificados.\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "[MonsterCore v2] RANSAC Multi-Plano: "
              << planes_found << " planos encontrados e retificados.\n";

    return result;
}

// ------ LAPLACIAN SMOOTH COM KERNEL NATIVO ------

// Constrói adjacência CSR a partir de faces (executado na CPU, uma vez)
std::pair<torch::Tensor, torch::Tensor> build_adjacency_csr(
    torch::Tensor faces_cpu, int num_verts
) {
    auto faces_acc = faces_cpu.accessor<int64_t, 2>();
    int num_faces = faces_cpu.size(0);

    // Construir lista de adjacência
    std::vector<std::vector<int>> adj(num_verts);
    for (int f = 0; f < num_faces; ++f) {
        int v0 = faces_acc[f][0], v1 = faces_acc[f][1], v2 = faces_acc[f][2];
        adj[v0].push_back(v1); adj[v0].push_back(v2);
        adj[v1].push_back(v0); adj[v1].push_back(v2);
        adj[v2].push_back(v0); adj[v2].push_back(v1);
    }

    // Deduplicar e construir CSR
    std::vector<int> offsets(num_verts + 1, 0);
    std::vector<int> indices;

    for (int v = 0; v < num_verts; ++v) {
        // Deduplicar vizinhos
        auto& neighbors = adj[v];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        offsets[v + 1] = offsets[v] + static_cast<int>(neighbors.size());
        for (int n : neighbors) indices.push_back(n);
    }

    auto offsets_tensor = torch::from_blob(offsets.data(), {num_verts + 1},
        torch::kInt32).clone();
    auto indices_tensor = torch::from_blob(indices.data(),
        {static_cast<int64_t>(indices.size())}, torch::kInt32).clone();

    return {offsets_tensor, indices_tensor};
}

torch::Tensor cuda_laplacian_smooth(
    torch::Tensor vertices,
    torch::Tensor faces,
    int iterations,
    float lambda_factor
) {
    TORCH_CHECK(vertices.is_cuda(), "Vértices devem estar na GPU.");
    TORCH_CHECK(faces.is_cuda(), "Faces devem estar na GPU.");

    int num_verts = vertices.size(0);
    auto v_in = vertices.to(torch::kFloat32).contiguous();
    auto v_out = torch::zeros_like(v_in);

    // Construir adjacência CSR na CPU (uma vez)
    auto faces_cpu = faces.to(torch::kLong).cpu();
    auto [offsets_cpu, indices_cpu] = build_adjacency_csr(faces_cpu, num_verts);

    // Mover para GPU
    auto offsets_gpu = offsets_cpu.to(torch::kCUDA);
    auto indices_gpu = indices_cpu.to(torch::kCUDA);

    // Criar stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int blocks = (num_verts + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < iterations; ++iter) {
        laplacian_smooth_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            v_in.data_ptr<float>(),
            v_out.data_ptr<float>(),
            offsets_gpu.data_ptr<int>(),
            indices_gpu.data_ptr<int>(),
            lambda_factor,
            num_verts
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Swap buffers (ping-pong)
        std::swap(v_in, v_out);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "[MonsterCore v2] CUDA Laplacian: "
              << iterations << " iterações em " << num_verts << " vértices (kernel nativo).\n";

    return v_in;  // Após número par de swaps, resultado está em v_in
}

// ------ QUADRIC EDGE COLLAPSE (DECIMATION NATIVA) ------

std::pair<torch::Tensor, torch::Tensor> cuda_quadric_decimate(
    torch::Tensor vertices,
    torch::Tensor faces,
    int target_faces
) {
    TORCH_CHECK(vertices.is_cuda(), "Vértices devem estar na GPU.");
    TORCH_CHECK(faces.is_cuda(), "Faces devem estar na GPU.");

    int num_verts = vertices.size(0);
    int num_faces = faces.size(0);

    if (num_faces <= target_faces) {
        std::cout << "[MonsterCore v2] Decimation: Mesh já está abaixo do target ("
                  << num_faces << " <= " << target_faces << ").\n";
        return {vertices, faces};
    }

    auto verts_f = vertices.to(torch::kFloat32).contiguous();
    auto faces_i = faces.to(torch::kInt32).contiguous();

    // Fase 1: Computar Quadrics por vértice via kernel
    auto quadrics = torch::zeros({num_verts, 10},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    int face_blocks = (num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_face_quadrics_kernel<<<face_blocks, BLOCK_SIZE>>>(
        verts_f.data_ptr<float>(),
        faces_i.data_ptr<int>(),
        quadrics.data_ptr<float>(),
        num_faces
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fase 2: Extrair arestas únicas
    auto f_cpu = faces.to(torch::kLong).cpu();
    auto f_acc = f_cpu.accessor<int64_t, 2>();

    std::set<std::pair<int,int>> edge_set;
    for (int f = 0; f < num_faces; ++f) {
        int v0 = f_acc[f][0], v1 = f_acc[f][1], v2 = f_acc[f][2];
        auto add_edge = [&](int a, int b) {
            if (a > b) std::swap(a, b);
            edge_set.insert({a, b});
        };
        add_edge(v0, v1); add_edge(v1, v2); add_edge(v0, v2);
    }

    int num_edges = static_cast<int>(edge_set.size());
    auto edges_cpu = torch::zeros({num_edges, 2}, torch::kInt32);
    int ei = 0;
    for (auto& [a, b] : edge_set) {
        edges_cpu[ei][0] = a;
        edges_cpu[ei][1] = b;
        ei++;
    }
    auto edges_gpu = edges_cpu.to(torch::kCUDA);

    // Fase 3: Computar custo de cada aresta via kernel
    auto costs = torch::zeros({num_edges},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto optimal_positions = torch::zeros({num_edges, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    int edge_blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_edge_collapse_cost_kernel<<<edge_blocks, BLOCK_SIZE>>>(
        verts_f.data_ptr<float>(),
        quadrics.data_ptr<float>(),
        edges_gpu.data_ptr<int>(),
        costs.data_ptr<float>(),
        optimal_positions.data_ptr<float>(),
        num_edges
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fase 4: Greedy Collapse (CPU — topologia é sequencial)
    auto costs_cpu = costs.cpu();
    auto opt_pos_cpu = optimal_positions.cpu();
    auto verts_cpu = verts_f.cpu();
    auto faces_cpu_long = f_cpu.clone();

    // Ordenar arestas por custo
    auto sorted_indices = torch::argsort(costs_cpu);
    auto sorted_acc = sorted_indices.accessor<int64_t, 1>();
    auto edges_acc = edges_cpu.accessor<int, 2>();
    auto verts_acc = verts_cpu.accessor<float, 2>();
    auto opt_acc = opt_pos_cpu.accessor<float, 2>();

    // Mapa de remoção de vértices
    std::vector<int> vertex_map(num_verts);
    for (int i = 0; i < num_verts; ++i) vertex_map[i] = i;

    auto find_root = [&](int v) -> int {
        while (vertex_map[v] != v) {
            vertex_map[v] = vertex_map[vertex_map[v]]; // Path compression
            v = vertex_map[v];
        }
        return v;
    };

    int faces_removed = 0;
    int target_removals = num_faces - target_faces;

    for (int si = 0; si < num_edges && faces_removed < target_removals; ++si) {
        int eidx = sorted_acc[si];
        int va = find_root(edges_acc[eidx][0]);
        int vb = find_root(edges_acc[eidx][1]);

        if (va == vb) continue; // Já colapsados

        // Colapsar vb → va
        vertex_map[vb] = va;

        // Mover va para posição ótima
        verts_acc[va][0] = opt_acc[eidx][0];
        verts_acc[va][1] = opt_acc[eidx][1];
        verts_acc[va][2] = opt_acc[eidx][2];

        faces_removed += 2; // Em média, cada collapse remove ~2 faces
    }

    // Rebuild faces com vertex_map
    auto faces_acc2 = faces_cpu_long.accessor<int64_t, 2>();
    std::vector<int64_t> new_faces_data;

    for (int f = 0; f < num_faces; ++f) {
        int v0 = find_root(faces_acc2[f][0]);
        int v1 = find_root(faces_acc2[f][1]);
        int v2 = find_root(faces_acc2[f][2]);

        // Remover faces degeneradas
        if (v0 == v1 || v1 == v2 || v0 == v2) continue;

        new_faces_data.push_back(v0);
        new_faces_data.push_back(v1);
        new_faces_data.push_back(v2);
    }

    int new_num_faces = static_cast<int>(new_faces_data.size()) / 3;
    auto new_faces = torch::from_blob(new_faces_data.data(),
        {new_num_faces, 3}, torch::kLong).clone().to(torch::kCUDA);
    auto new_verts = verts_cpu.to(torch::kCUDA);

    std::cout << "[MonsterCore v2] Decimation: " << num_faces
              << " → " << new_num_faces << " faces ("
              << static_cast<int>((1.0f - static_cast<float>(new_num_faces) / num_faces) * 100)
              << "% redução).\n";

    return {new_verts, new_faces};
}

// ------ PINNED MEMORY TRANSFER ------

torch::Tensor pinned_to_gpu(torch::Tensor cpu_tensor) {
    TORCH_CHECK(!cpu_tensor.is_cuda(), "Tensor já está na GPU.");

    // Alocar memória pinned (page-locked)
    auto options = torch::TensorOptions()
        .dtype(cpu_tensor.dtype())
        .device(torch::kCPU)
        .pinned_memory(true);
    auto pinned = torch::empty_like(cpu_tensor, options);
    pinned.copy_(cpu_tensor);

    // Transfer assíncrono para GPU
    return pinned.to(torch::kCUDA, /*non_blocking=*/true);
}

torch::Tensor gpu_to_pinned(torch::Tensor gpu_tensor) {
    TORCH_CHECK(gpu_tensor.is_cuda(), "Tensor deve estar na GPU.");

    auto options = torch::TensorOptions()
        .dtype(gpu_tensor.dtype())
        .device(torch::kCPU)
        .pinned_memory(true);
    auto pinned = torch::empty_like(gpu_tensor, options);
    pinned.copy_(gpu_tensor, /*non_blocking=*/true);

    return pinned;
}
