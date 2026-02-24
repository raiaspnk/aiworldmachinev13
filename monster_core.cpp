// ============================================================================
//  MONSTER_CORE.CPP – Zero-Copy C++/CUDA Engine para o AI World Engine
// ============================================================================
//
//  Este módulo substitui os gargalos críticos do Python por código nativo:
//    1. MonsterPool   – Arena Allocator estático na VRAM (substitui empty_cache)
//    2. Zero-Copy Tiling – Fatiamento de imagens via views de tensor (0 cópia)
//    3. GPU RANSAC    – Retificação de planos Hard-Surface hiper-paralela
//    4. GPU Laplacian – Suavização de mesh direto na VRAM
//    5. Async Pipeline – Orquestrador multi-thread bypassando o GIL do Python
//
//  Build: python setup.py build_ext --inplace
//  Uso:   import monster_core
//
// ============================================================================

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <vector>
#include <mutex>
#include <future>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "monster_core_kernels.h"


namespace py = pybind11;

// ============================================================================
// 1. STATIC MEMORY ARENA (MonsterPool)
// ============================================================================
//
// Problema: torch.cuda.empty_cache() no Python é O(N) e pode levar 100ms+
//           entre etapas do pipeline, causando "soluços" de frame timing.
//
// Solução:  Pré-alocamos um bloco contínuo de VRAM no boot.
//           reset() é O(1) — apenas move um ponteiro.
//           Elimina chamadas ao cudaFree/cudaMalloc entre etapas.
//
// ============================================================================

class MonsterPool {
private:
    torch::Tensor arena_;
    size_t capacity_bytes_;
    size_t current_offset_;
    std::mutex pool_mutex_;

public:
    MonsterPool(size_t size_mb) {
        capacity_bytes_ = size_mb * 1024 * 1024;
        // Pré-aloca bloco contínuo na VRAM
        arena_ = torch::empty(
            {static_cast<int64_t>(capacity_bytes_)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );
        current_offset_ = 0;
        std::cout << "[MonsterCore] Arena inicializada: "
                  << size_mb << " MB na VRAM ("
                  << capacity_bytes_ << " bytes)\n";
    }

    // Aloca um tensor apontando para a arena pré-alocada (zero-copy)
    torch::Tensor allocate_tensor(std::vector<int64_t> dims, torch::ScalarType dtype) {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        size_t element_size = torch::elementSize(dtype);
        size_t total_elements = 1;
        for (auto d : dims) total_elements *= static_cast<size_t>(d);
        size_t requested_bytes = total_elements * element_size;

        // Alinhamento a 256 bytes (requisito CUDA para coalesced access)
        size_t aligned_bytes = ((requested_bytes + 255) / 256) * 256;

        if (current_offset_ + aligned_bytes > capacity_bytes_) {
            throw std::runtime_error(
                "[MonsterCore] OOM na Arena! "
                "Solicitado: " + std::to_string(aligned_bytes / (1024*1024)) + " MB, "
                "Disponível: " + std::to_string((capacity_bytes_ - current_offset_) / (1024*1024)) + " MB. "
                "Aumente o tamanho do MonsterPool."
            );
        }

        auto slice = arena_.slice(0, current_offset_, current_offset_ + requested_bytes);
        current_offset_ += aligned_bytes;

        return torch::from_blob(
            slice.data_ptr(), dims,
            torch::TensorOptions().dtype(dtype).device(torch::kCUDA)
        );
    }

    // Retorna bytes usados
    size_t used_bytes() const { return current_offset_; }
    size_t capacity() const { return capacity_bytes_; }
    double usage_percent() const {
        return (static_cast<double>(current_offset_) / capacity_bytes_) * 100.0;
    }

    // "Limpa" a memória em O(1) — apenas reseta o ponteiro
    void reset() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        current_offset_ = 0;
    }
};

// Instância global do pool
static std::unique_ptr<MonsterPool> global_pool = nullptr;

void init_pool(int size_mb) {
    global_pool = std::make_unique<MonsterPool>(static_cast<size_t>(size_mb));
}

void reset_pool() {
    if (global_pool) {
        global_pool->reset();
        std::cout << "[MonsterCore] Arena resetada em O(1).\n";
    }
}

torch::Tensor pool_allocate(std::vector<int64_t> dims, int dtype_int) {
    TORCH_CHECK(global_pool != nullptr, "MonsterPool não inicializado! Chame init_pool() primeiro.");
    auto dtype = static_cast<torch::ScalarType>(dtype_int);
    return global_pool->allocate_tensor(dims, dtype);
}

std::string pool_status() {
    if (!global_pool) return "Pool não inicializado.";
    return "[MonsterCore] Arena: "
         + std::to_string(global_pool->used_bytes() / (1024*1024)) + "/"
         + std::to_string(global_pool->capacity() / (1024*1024)) + " MB ("
         + std::to_string(static_cast<int>(global_pool->usage_percent())) + "%)";
}

// ============================================================================
// 2. ZERO-COPY GPU TILING
// ============================================================================
//
// Problema: O World Tiling em Python fazia crop → save → reload por tile.
//           Isso gera N cópias CPU↔GPU e N escritas em disco.
//
// Solução:  Retorna views (slices) do tensor CUDA original.
//           Zero cópias de memória. Zero I/O de disco.
//
// ============================================================================

std::vector<torch::Tensor> zero_copy_tiling(
    torch::Tensor master_img, int grid_size, int overlap_px
) {
    TORCH_CHECK(master_img.is_cuda(), "Imagem deve estar na GPU (CUDA tensor).");
    TORCH_CHECK(master_img.dim() == 3 || master_img.dim() == 4,
                "Formato esperado: [C, H, W] ou [B, C, H, W]");

    // Se batch, pegar a primeira imagem
    if (master_img.dim() == 4) {
        master_img = master_img.squeeze(0);
    }

    int h = master_img.size(1);  // [C, H, W]
    int w = master_img.size(2);
    int tile_h = h / grid_size;
    int tile_w = w / grid_size;

    std::vector<torch::Tensor> tiles;
    tiles.reserve(grid_size * grid_size);

    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            int y1 = std::max(0, row * tile_h - overlap_px);
            int y2 = std::min(h, (row + 1) * tile_h + overlap_px);
            int x1 = std::max(0, col * tile_w - overlap_px);
            int x2 = std::min(w, (col + 1) * tile_w + overlap_px);

            // Fatiamento direto na VRAM — view, NÃO cópia
            auto tile = master_img.index({
                torch::indexing::Slice(),                  // Canal
                torch::indexing::Slice(y1, y2),            // Altura
                torch::indexing::Slice(x1, x2)             // Largura
            });
            tiles.push_back(tile);
        }
    }

    std::cout << "[MonsterCore] Tiling: " << tiles.size()
              << " tiles (" << grid_size << "x" << grid_size
              << ") — Zero cópias de memória.\n";

    return tiles;
}

// ============================================================================
// GPU GEOMETRY (Implemented in monster_core_kernels.cu)
// ============================================================================
// Now using Warp Reductions and Shared Memory in CUDA for 10x-100x performance


// ============================================================================
// 5. ASYNC GEOMETRY PIPELINE (GIL Bypass)
// ============================================================================
//
// Orquestra tiling + geometria em threads C++ nativas,
// sem a trava do interpretador Python (GIL).
//
// FIX: Captura tensores por VALOR (não referência) para evitar
//      race conditions quando o Python garbage-collecta os originais.
//
// ============================================================================

py::dict async_geometry_pipeline(
    torch::Tensor master_img,
    torch::Tensor vertices,
    torch::Tensor faces,
    int tile_grid,
    int overlap_px,
    float ransac_threshold,
    int ransac_iters,
    int smooth_iters,
    float smooth_lambda
) {
    // Clonar tensores para ownership segura nas threads
    auto img_copy = master_img.clone();
    auto verts_copy = vertices.clone();
    auto faces_copy = faces.clone();

    // Resultados compartilhados
    std::vector<torch::Tensor> tile_results;
    torch::Tensor rectified_result;
    torch::Tensor smoothed_result;
    std::mutex result_mutex;
    bool tiling_ok = false;
    bool geometry_ok = false;

    {
        // Libera o GIL do Python — daqui em diante é C++ puro
        py::gil_scoped_release release;

        // Thread 1: Zero-Copy Tiling
        std::jthread t1([&tile_results, &tiling_ok, &result_mutex,
                         img_copy, tile_grid, overlap_px]() {
            try {
                auto tiles = zero_copy_tiling(img_copy, tile_grid, overlap_px);
                std::lock_guard<std::mutex> lock(result_mutex);
                tile_results = std::move(tiles);
                tiling_ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[MonsterCore] Tiling thread error: " << e.what() << "\n";
            }
        });

        // Thread 2: RANSAC + Laplacian Smooth
        std::jthread t2([&rectified_result, &smoothed_result, &geometry_ok, &result_mutex,
                         verts_copy, faces_copy, ransac_threshold, ransac_iters,
                         smooth_iters, smooth_lambda]() {
            try {
                auto rectified = launch_gpu_ransac_hard_surface(
                    verts_copy, ransac_threshold, ransac_iters, /*batch_size=*/100
                );
                auto smoothed = launch_gpu_laplacian_smooth(
                    rectified, faces_copy, smooth_iters, smooth_lambda
                );
                
                // Return explicitly via Pinned Memory for faster transfer back to CPU
                smoothed = smoothed.to(torch::TensorOptions().device(torch::kCPU).pinned_memory(true));

                std::lock_guard<std::mutex> lock(result_mutex);
                rectified_result = rectified;
                smoothed_result = smoothed;
                geometry_ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[MonsterCore] Geometry thread error: " << e.what() << "\n";
            }
        });

        // std::jthread faz join automaticamente ao sair do escopo
    }
    // GIL re-adquirido aqui automaticamente

    // Montar resultado como dict Python
    py::dict result;
    result["tiling_ok"] = tiling_ok;
    result["geometry_ok"] = geometry_ok;

    if (tiling_ok) {
        py::list tile_list;
        for (auto& t : tile_results) tile_list.append(t);
        result["tiles"] = tile_list;
        result["num_tiles"] = static_cast<int>(tile_results.size());
    }

    if (geometry_ok) {
        result["rectified_vertices"] = rectified_result;
        result["smoothed_vertices"] = smoothed_result;
    }

    return result;
}

// ============================================================================
// PYBIND11 BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MonsterCore – Zero-Copy C++/CUDA Engine para AI World Engine";

    // Arena Allocator
    m.def("init_pool", &init_pool,
          "Inicializa a Arena de Memória Estática (tamanho em MB)",
          py::arg("size_mb") = 8192);
    m.def("reset_pool", &reset_pool,
          "Esvazia a Arena em O(1) sem chamar cudaFree");
    m.def("pool_allocate", &pool_allocate,
          "Aloca tensor na Arena pré-alocada (zero malloc)",
          py::arg("dims"), py::arg("dtype_int"));
    m.def("pool_status", &pool_status,
          "Retorna status de uso da Arena");

    // Tiling
    m.def("zero_copy_tiling", &zero_copy_tiling,
          "Retorna views fatiados do tensor na VRAM (zero cópia)",
          py::arg("master_img"), py::arg("grid_size") = 2, py::arg("overlap_px") = 64);

    // Geometria
    m.def("gpu_ransac_hard_surface", &launch_gpu_ransac_hard_surface,
          "Retificação de planos Hard-Surface via RANSAC paralelo na GPU (Kernel CUDA)",
          py::arg("vertices"), py::arg("distance_threshold") = 0.02f,
          py::arg("num_iterations") = 1000, py::arg("batch_size") = 100);
    m.def("gpu_laplacian_smooth", &launch_gpu_laplacian_smooth,
          "Suavização Laplaciana na GPU via Caching & CSR (Kernel CUDA)",
          py::arg("vertices"), py::arg("faces"),
          py::arg("iterations") = 5, py::arg("lambda_factor") = 0.5f);

    // Pipeline Assíncrono
    m.def("async_geometry_pipeline", &async_geometry_pipeline,
          "Pipeline completo: Tiling + RANSAC + Smooth em paralelo (bypassa GIL)",
          py::arg("master_img"), py::arg("vertices"), py::arg("faces"),
          py::arg("tile_grid") = 2, py::arg("overlap_px") = 64,
          py::arg("ransac_threshold") = 0.02f, py::arg("ransac_iters") = 1000,
          py::arg("smooth_iters") = 5, py::arg("smooth_lambda") = 0.5f);
}
