#include "monster_core_kernels.h"
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// 1. RANSAC KERNEL (Warp Shuffle Reduction)
// ============================================================================

__global__ void ransac_evaluate_planes_kernel(
    const float* __restrict__ vertices,
    const float* __restrict__ plane_points,
    const float* __restrict__ plane_normals,
    const bool* __restrict__ valid_planes,
    int* __restrict__ inlier_counts,
    float distance_threshold,
    int N,
    int batch_size) 
{
    // Grid: ( (N + 255)/256, batch_size )
    int batch_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_idx = threadIdx.x % 32;

    if (!valid_planes[batch_idx]) return;

    // Load plane equation into registers (broadcast to all threads in block)
    float px = plane_points[batch_idx * 3 + 0];
    float py = plane_points[batch_idx * 3 + 1];
    float pz = plane_points[batch_idx * 3 + 2];
    
    float nx = plane_normals[batch_idx * 3 + 0];
    float ny = plane_normals[batch_idx * 3 + 1];
    float nz = plane_normals[batch_idx * 3 + 2];

    int is_inlier = 0;

    if (point_idx < N) {
        float vx = vertices[point_idx * 3 + 0];
        float vy = vertices[point_idx * 3 + 1];
        float vz = vertices[point_idx * 3 + 2];

        float dist = fabsf((vx - px) * nx + (vy - py) * ny + (vz - pz) * nz);
        if (dist < distance_threshold) {
            is_inlier = 1;
        }
    }

    // Warp Reduction (Warp-Level Primitives vs Global Atomics)
    unsigned int mask = 0xffffffff;
    int warp_sum = is_inlier;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(mask, warp_sum, offset);
    }

    // Only lane 0 of each warp adds to the global counter
    if (lane_idx == 0 && warp_sum > 0) {
        atomicAdd(&inlier_counts[batch_idx], warp_sum);
    }
}

torch::Tensor launch_gpu_ransac_hard_surface(
    torch::Tensor vertices,
    float distance_threshold,
    int num_iterations,
    int batch_size)
{
    int N = vertices.size(0);
    
    int64_t global_best_count = 0;
    torch::Tensor global_best_normal = torch::zeros({3}, vertices.options());
    torch::Tensor global_best_point = torch::zeros({3}, vertices.options());
    
    auto opts = vertices.options();
    auto opts_int = opts.dtype(torch::kInt32);
    auto opts_bool = opts.dtype(torch::kBool);
    
    for (int b = 0; b < num_iterations; b += batch_size) {
        int current_batch = std::min(batch_size, num_iterations - b);
        
        // Randomly select 3 points per plane
        auto indices = torch::randint(0, N, {current_batch, 3}, opts.dtype(torch::kLong));
        auto idx_flat = indices.flatten();
        auto pts = vertices.index_select(0, idx_flat).view({current_batch, 3, 3});
        
        auto p1 = pts.select(1, 0);
        auto p2 = pts.select(1, 1);
        auto p3 = pts.select(1, 2);
        
        auto v1 = p2 - p1;
        auto v2 = p3 - p1;
        auto normals = torch::cross(v1, v2, 1);
        auto norms = normals.norm(2, 1, true);
        
        auto valid_mask = (norms.squeeze(1) > 1e-8f);
        if (valid_mask.sum().item<int>() == 0) continue;
        
        normals = normals / (norms + 1e-10f);
        
        // Prepare global counters
        auto inlier_counts = torch::zeros({current_batch}, opts_int);
        
        // Launch Configuration
        dim3 threads(256);
        dim3 blocks((N + 255) / 256, current_batch);
        
        ransac_evaluate_planes_kernel<<<blocks, threads>>>(
            vertices.data_ptr<float>(),
            p1.contiguous().data_ptr<float>(),
            normals.contiguous().data_ptr<float>(),
            valid_mask.contiguous().data_ptr<bool>(),
            inlier_counts.data_ptr<int>(),
            distance_threshold,
            N,
            current_batch
        );
        
        // Find local best
        int best_local_idx = torch::argmax(inlier_counts).item<int>();
        int best_local_count = inlier_counts[best_local_idx].item<int>();
        
        if (best_local_count > global_best_count) {
            global_best_count = best_local_count;
            global_best_normal = normals[best_local_idx].clone();
            global_best_point = p1[best_local_idx].clone();
        }
    }
    
    if (global_best_count == 0) return vertices;
    
    auto rectified = vertices.clone();
    
    // Project inliers in PyTorch (fast enough since it's just 1 operation)
    auto diff = rectified - global_best_point.unsqueeze(0);
    auto distances = torch::sum(diff * global_best_normal.unsqueeze(0), 1, true);
    auto mask = torch::abs(distances).squeeze(1) < distance_threshold;
    
    auto inlier_indices = torch::nonzero(mask).squeeze(1);
    auto inlier_verts = rectified.index_select(0, inlier_indices);
    auto diff_to_plane = inlier_verts - global_best_point.unsqueeze(0);
    auto proj_dist = torch::sum(diff_to_plane * global_best_normal.unsqueeze(0), 1, true);
    auto corrected = inlier_verts - proj_dist * global_best_normal.unsqueeze(0);
    
    rectified.index_copy_(0, inlier_indices, corrected);
    return rectified;
}

// ============================================================================
// 2. LAPLACIAN SMOOTH KERNEL (Shared Memory + CSR Graph)
// ============================================================================

__global__ void build_csr_degrees(
    const int64_t* __restrict__ faces,
    int* __restrict__ node_counts,
    int F) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < F) {
        int v0 = faces[idx * 3 + 0];
        int v1 = faces[idx * 3 + 1];
        int v2 = faces[idx * 3 + 2];
        
        atomicAdd(&node_counts[v0], 2);
        atomicAdd(&node_counts[v1], 2);
        atomicAdd(&node_counts[v2], 2);
    }
}

__global__ void build_csr_neighbors(
    const int64_t* __restrict__ faces,
    const int* __restrict__ node_offsets,
    int* __restrict__ current_offsets,
    int* __restrict__ neighbors,
    int F) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < F) {
        int v[3] = {
            (int)faces[idx * 3 + 0],
            (int)faces[idx * 3 + 1],
            (int)faces[idx * 3 + 2]
        };
        
        for (int i=0; i<3; ++i) {
            int src = v[i];
            int dst1 = v[(i+1)%3];
            int dst2 = v[(i+2)%3];
            
            int offset1 = atomicAdd(&current_offsets[src], 1);
            neighbors[node_offsets[src] + offset1] = dst1;
            
            int offset2 = atomicAdd(&current_offsets[src], 1);
            neighbors[node_offsets[src] + offset2] = dst2;
        }
    }
}

__global__ void laplacian_smooth_csr_kernel(
    const float* __restrict__ vertices_in,
    float* __restrict__ vertices_out,
    const int* __restrict__ node_offsets,
    const int* __restrict__ node_counts,
    const int* __restrict__ neighbors,
    float lambda_factor,
    int N)
{
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= N) return;
    
    // Shared memory optimization: Load the current vertex into register/shared memory
    float vx = vertices_in[v_idx * 3 + 0];
    float vy = vertices_in[v_idx * 3 + 1];
    float vz = vertices_in[v_idx * 3 + 2];
    
    int count = node_counts[v_idx];
    if (count == 0) {
        vertices_out[v_idx * 3 + 0] = vx;
        vertices_out[v_idx * 3 + 1] = vy;
        vertices_out[v_idx * 3 + 2] = vz;
        return;
    }
    
    int offset = node_offsets[v_idx];
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    int valid_neighbors = 0;
    
    // Bilateral Threshold (Anti-Melting)
    // Se a diferença do Z for maior que X (ex: 0.05), é uma quina de prédio/muro, não suavize com esse vizinho.
    float bilateral_threshold = 0.02f; // Pode ser passado como parâmetro futuramente, fixado para cenários
    
    for (int i = 0; i < count; ++i) {
        int n_idx = neighbors[offset + i];
        
        float n_vz = __ldg(&vertices_in[n_idx * 3 + 2]);
        if (fabsf(n_vz - vz) < bilateral_threshold) {
            sum_x += __ldg(&vertices_in[n_idx * 3 + 0]);
            sum_y += __ldg(&vertices_in[n_idx * 3 + 1]);
            sum_z += n_vz;
            valid_neighbors++;
        }
    }
    
    if (valid_neighbors == 0) {
        vertices_out[v_idx * 3 + 0] = vx;
        vertices_out[v_idx * 3 + 1] = vy;
        vertices_out[v_idx * 3 + 2] = vz;
        return;
    }
    
    float mean_x = sum_x / (float)valid_neighbors;
    float mean_y = sum_y / (float)valid_neighbors;
    float mean_z = sum_z / (float)valid_neighbors;
    
    vertices_out[v_idx * 3 + 0] = (1.0f - lambda_factor) * vx + lambda_factor * mean_x;
    vertices_out[v_idx * 3 + 1] = (1.0f - lambda_factor) * vy + lambda_factor * mean_y;
    vertices_out[v_idx * 3 + 2] = (1.0f - lambda_factor) * vz + lambda_factor * mean_z;
}

torch::Tensor launch_gpu_laplacian_smooth(
    torch::Tensor vertices,
    torch::Tensor faces,
    int iterations,
    float lambda_factor)
{
    int N = vertices.size(0);
    int F = faces.size(0);
    
    auto opts_int = vertices.options().dtype(torch::kInt32);
    
    auto node_counts = torch::zeros({N}, opts_int);
    
    int threads_f = 256;
    int blocks_f = (F + threads_f - 1) / threads_f;
    
    // 1. Build Degrees
    build_csr_degrees<<<blocks_f, threads_f>>>(
        faces.data_ptr<int64_t>(),
        node_counts.data_ptr<int>(),
        F
    );
    
    // 2. Prefix Sum for Offsets (using ATen since it's highly optimized)
    auto node_offsets = torch::zeros({N + 1}, opts_int);
    node_offsets.slice(0, 1) = node_counts.cumsum(0, torch::kInt32);
    
    int total_edges = node_offsets[N].item<int>();
    auto neighbors = torch::empty({total_edges}, opts_int);
    auto current_offsets = torch::zeros({N}, opts_int);
    
    // 3. Populate Neighbors
    build_csr_neighbors<<<blocks_f, threads_f>>>(
        faces.data_ptr<int64_t>(),
        node_offsets.data_ptr<int>(),
        current_offsets.data_ptr<int>(),
        neighbors.data_ptr<int>(),
        F
    );
    
    // 4. Smoothing Iterations
    int threads_v = 256;
    int blocks_v = (N + threads_v - 1) / threads_v;
    
    auto v_in = vertices.clone().to(torch::kFloat32);
    auto v_out = torch::empty_like(v_in);
    
    for (int iter = 0; iter < iterations; ++iter) {
        laplacian_smooth_csr_kernel<<<blocks_v, threads_v>>>(
            v_in.data_ptr<float>(),
            v_out.data_ptr<float>(),
            node_offsets.data_ptr<int>(),
            node_counts.data_ptr<int>(),
            neighbors.data_ptr<int>(),
            lambda_factor,
            N
        );
        std::swap(v_in, v_out);
    }
    
    return ((iterations % 2) == 1) ? v_out : v_in;
}

// ============================================================================
// 3. TERRAIN GENERATION KERNEL (Depth Displacement)
// ============================================================================

__global__ void generate_displaced_vertices_kernel(
    const float* __restrict__ depth_map,
    float* __restrict__ vertices,
    int res,
    float max_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = res * res;
    if (idx < max_idx) {
        int row = idx / res;
        int col = idx % res;
        
        float x = (float)col / (res - 1) - 0.5f;
        float y = (float)row / (res - 1) - 0.5f;
        float z = depth_map[idx] * max_height;
        
        vertices[idx * 3 + 0] = x;
        vertices[idx * 3 + 1] = y;
        vertices[idx * 3 + 2] = z;
    }
}

__global__ void generate_grid_faces_kernel(
    int64_t* __restrict__ faces,
    int res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = (res - 1) * (res - 1);
    if (idx < max_idx) {
        int row = idx / (res - 1);
        int col = idx % (res - 1);
        
        int v00 = row * res + col;
        int v10 = (row + 1) * res + col;
        int v01 = row * res + (col + 1);
        int v11 = (row + 1) * res + (col + 1);
        
        // Triangle 1: (v00, v10, v01)
        faces[idx * 6 + 0] = v00;
        faces[idx * 6 + 1] = v10;
        faces[idx * 6 + 2] = v01;
        
        // Triangle 2: (v10, v11, v01)
        faces[idx * 6 + 3] = v10;
        faces[idx * 6 + 4] = v11;
        faces[idx * 6 + 5] = v01;
    }
}

std::vector<torch::Tensor> launch_gpu_generate_displaced_grid(
    torch::Tensor depth_map,
    float max_height)
{
    TORCH_CHECK(depth_map.is_cuda(), "Depth map must be a CUDA tensor.");
    TORCH_CHECK(depth_map.dim() == 2, "Depth map must be 2D [H, W].");
    TORCH_CHECK(depth_map.size(0) == depth_map.size(1), "Depth map must be square for grid generation.");
    
    int res = depth_map.size(0);
    
    auto opts_float = depth_map.options();
    auto opts_long = depth_map.options().dtype(torch::kInt64);
    
    auto vertices = torch::empty({res * res, 3}, opts_float);
    auto faces = torch::empty({(res - 1) * (res - 1) * 2, 3}, opts_long);
    
    int num_vertices = res * res;
    int num_quads = (res - 1) * (res - 1);
    
    int threads_v = 256;
    int blocks_v = (num_vertices + threads_v - 1) / threads_v;
    generate_displaced_vertices_kernel<<<blocks_v, threads_v>>>(
        depth_map.contiguous().data_ptr<float>(),
        vertices.data_ptr<float>(),
        res,
        max_height
    );
    
    int threads_f = 256;
    int blocks_f = (num_quads + threads_f - 1) / threads_f;
    generate_grid_faces_kernel<<<blocks_f, threads_f>>>(
        faces.data_ptr<int64_t>(),
        res
    );
    
    // Sincroniza kernel para checar erros
    cudaDeviceSynchronize();
    
    return {vertices, faces};
}
