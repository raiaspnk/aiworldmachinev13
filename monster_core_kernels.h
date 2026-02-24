#pragma once

#include <torch/extension.h>

// ============================================================================
// MonsterCore v2 - Pure CUDA Kernels
// ============================================================================

// Launches the RANSAC kernel optimized with warp reductions
torch::Tensor launch_gpu_ransac_hard_surface(
    torch::Tensor vertices,
    float distance_threshold,
    int num_iterations,
    int batch_size
);

// Launches the Laplacian Smooth kernel optimized with shared memory
torch::Tensor launch_gpu_laplacian_smooth(
    torch::Tensor vertices,
    torch::Tensor faces,
    int iterations,
    float lambda_factor
);
