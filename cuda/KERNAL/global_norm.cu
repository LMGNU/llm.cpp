#include "../includes/global_norm.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

namespace quadtrix {
namespace cuda {
namespace {

constexpr int kNormBlockSize = 256;

bool valid_f32_cuda(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.is_contiguous();
}

__device__ float block_sum(float value, float* shared) {
    value = warp_sum(value);
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;
    value = threadIdx.x < warp_count ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__global__ void norm_squared_kernel(const float* __restrict__ grads, float* __restrict__ partial_sums, std::size_t n) {
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float grad = grads[idx];
        local_sum += grad * grad;
    }
    const float block_total = block_sum(local_sum, shared);
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_total;
    }
}

__global__ void clip_kernel(float* grads, std::size_t n, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        grads[idx] *= scale;
    }
}

}  // namespace

Status global_norm_squared(const TensorView& grads, TensorView partial_sums, cudaStream_t stream) {
    if (!valid_f32_cuda(grads) || !valid_f32_cuda(partial_sums) || partial_sums.shape.rank != 1) {
        return Status::failure(cudaErrorInvalidValue, "invalid global_norm_squared arguments");
    }

    const int blocks = static_cast<int>(partial_sums.shape.dims[0]);
    if (blocks <= 0) {
        return Status::failure(cudaErrorInvalidValue, "partial_sums must have at least one element");
    }

    DeviceGuard guard(grads.device_id);
    const std::size_t shared_bytes = ((kNormBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    norm_squared_kernel<<<blocks, kNormBlockSize, shared_bytes, stream>>>(
        grads.data_as<const float>(),
        partial_sums.data_as<float>(),
        grads.numel());
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

Status clip_gradients_by_global_norm(TensorView grads, float global_norm, float max_norm, cudaStream_t stream) {
    if (!valid_f32_cuda(grads) || max_norm <= 0.0f) {
        return Status::failure(cudaErrorInvalidValue, "invalid clip_gradients_by_global_norm arguments");
    }

    DeviceGuard guard(grads.device_id);
    clip_kernel<<<one_dim_grid(grads.numel()), kDefaultBlockSize, 0, stream>>>(
        grads.data_as<float>(),
        grads.numel(),
        clip_scale(global_norm, max_norm));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
