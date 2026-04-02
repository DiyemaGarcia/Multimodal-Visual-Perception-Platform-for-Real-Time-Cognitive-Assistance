#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * elbo_kernel.cu
 *
 * Custom CUDA kernel for parallel computation of the KL divergence term
 * in the ELBO: KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
 *
 * Operates element-wise across (batch, latent_dim) tensors.
 * Result is summed over the latent dimension and returned per sample.
 */

__global__ void kl_divergence_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ logvar,
    float* __restrict__ kl_per_sample,
    int batch_size,
    int latent_dim
) {
    // One thread per (batch, latent_dim) element
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int d = blockIdx.y * blockDim.y + threadIdx.y;  // latent dim index

    if (b < batch_size && d < latent_dim) {
        int idx = b * latent_dim + d;
        float mu_val = mu[idx];
        float lv_val = logvar[idx];
        float kl_elem = -0.5f * (1.0f + lv_val - mu_val * mu_val - expf(lv_val));

        // Atomic add into per-sample accumulator
        atomicAdd(&kl_per_sample[b], kl_elem);
    }
}

/*
 * Python-callable KL divergence computation.
 *
 * Args:
 *   mu:     (batch, latent_dim) posterior means.
 *   logvar: (batch, latent_dim) posterior log-variances.
 *
 * Returns:
 *   kl: (batch,) per-sample KL divergence values.
 */
torch::Tensor compute_kl_divergence(
    torch::Tensor mu,
    torch::Tensor logvar
) {
    TORCH_CHECK(mu.is_cuda(), "mu must be a CUDA tensor");
    TORCH_CHECK(logvar.is_cuda(), "logvar must be a CUDA tensor");
    TORCH_CHECK(mu.sizes() == logvar.sizes(), "mu and logvar must have the same shape");
    TORCH_CHECK(mu.dtype() == torch::kFloat32, "Only float32 supported");

    int batch_size = mu.size(0);
    int latent_dim = mu.size(1);

    auto kl = torch::zeros({batch_size}, mu.options());

    dim3 block(16, 16);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (latent_dim + block.y - 1) / block.y
    );

    kl_divergence_kernel<<<grid, block>>>(
        mu.data_ptr<float>(),
        logvar.data_ptr<float>(),
        kl.data_ptr<float>(),
        batch_size,
        latent_dim
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA kernel failed: ", cudaGetErrorString(err));

    return kl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_kl_divergence",
        &compute_kl_divergence,
        "Fast CUDA KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).",
        py::arg("mu"), py::arg("logvar")
    );
}