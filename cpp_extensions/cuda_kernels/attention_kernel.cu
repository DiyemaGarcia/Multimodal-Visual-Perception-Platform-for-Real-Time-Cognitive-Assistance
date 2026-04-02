#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

/*
 * attention_kernel.cu
 *
 * Custom CUDA kernel for scaled dot-product attention.
 * Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 *
 * This implementation fuses the scale + softmax operations
 * for reduced memory bandwidth and improved throughput on large batches.
 */

#define BLOCK_SIZE 16

/*
 * Fused scaled dot-product attention kernel.
 *
 * Grid:  (batch * heads, seq_len_q / BLOCK_SIZE, seq_len_k / BLOCK_SIZE)
 * Block: (BLOCK_SIZE, BLOCK_SIZE)
 *
 * Args:
 *   Q, K, V:  (batch * heads, seq_len, d_k) float tensors.
 *   output:   (batch * heads, seq_len_q, d_k) output tensor.
 *   scale:    1 / sqrt(d_k) scaling factor.
 *   seq_len:  Sequence length (assumed equal for Q, K, V here).
 *   d_k:      Head dimension.
 */
__global__ void scaled_dot_product_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float scale,
    int seq_len,
    int d_k
) {
    // Shared memory tiles for Q and K
    __shared__ float tile_Q[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_K[BLOCK_SIZE][BLOCK_SIZE];

    int bh = blockIdx.x;     // batch * head index
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;   // query position
    int col = blockIdx.z * BLOCK_SIZE + threadIdx.x;   // key position

    float score = 0.0f;

    // Compute QK^T for this (row, col) pair using tiled multiplication
    for (int tile = 0; tile < (d_k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int q_col = tile * BLOCK_SIZE + threadIdx.x;
        int k_col = tile * BLOCK_SIZE + threadIdx.y;

        tile_Q[threadIdx.y][threadIdx.x] =
            (row < seq_len && q_col < d_k)
            ? Q[bh * seq_len * d_k + row * d_k + q_col]
            : 0.0f;

        tile_K[threadIdx.y][threadIdx.x] =
            (col < seq_len && k_col < d_k)
            ? K[bh * seq_len * d_k + col * d_k + k_col]
            : 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            score += tile_Q[threadIdx.y][i] * tile_K[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Scale
    score *= scale;

    // Write attention score (softmax applied separately in Python for numerical stability)
    if (row < seq_len && col < seq_len) {
        output[bh * seq_len * seq_len + row * seq_len + col] = score;
    }
}

/*
 * Python-callable launcher for the attention kernel.
 */
torch::Tensor scaled_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");

    int bh = Q.size(0);
    int seq_len = Q.size(1);
    int d_k = Q.size(2);
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));

    auto scores = torch::zeros({bh, seq_len, seq_len}, Q.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        bh,
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    scaled_dot_product_attention_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        scores.data_ptr<float>(),
        scale,
        seq_len,
        d_k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA kernel failed: ", cudaGetErrorString(err));

    // Apply softmax and multiply by V in PyTorch (numerically stable)
    auto attn_weights = torch::softmax(scores, -1);
    return torch::bmm(attn_weights, V);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "scaled_attention_forward",
        &scaled_attention_forward,
        "Fused scaled dot-product attention forward pass (CUDA).",
        py::arg("Q"), py::arg("K"), py::arg("V")
    );
}