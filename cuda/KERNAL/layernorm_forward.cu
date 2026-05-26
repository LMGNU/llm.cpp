#include "../includes/layernorm.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>
#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

constexpr int kLayerNormBlockSize = 256;

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

bool valid_vector(const TensorView& tensor, std::int64_t width) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.rank == 1 && tensor.shape.dims[0] == width && tensor.shape.is_contiguous();
}

bool valid_matrix_pair(const TensorView& input, const TensorView& output) {
    return input.data != nullptr && output.data != nullptr && input.device == DeviceKind::CUDA &&
           output.device == DeviceKind::CUDA && input.device_id == output.device_id &&
           input.dtype == DType::F32 && output.dtype == DType::F32 && input.shape.rank == 2 &&
           output.shape.rank == 2 && input.shape.is_contiguous() && output.shape.is_contiguous() &&
           input.shape.dims[0] == output.shape.dims[0] && input.shape.dims[1] == output.shape.dims[1];
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

__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int rows,
    int width,
    float epsilon) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* __restrict__ row_input = input + row * width;
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        local_sum += row_input[col];
    }

    const float row_sum = block_sum(local_sum, shared);
    const float row_mean = row_sum / static_cast<float>(width);

    float local_var_sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float shifted = row_input[col] - row_mean;
        local_var_sum += shifted * shifted;
    }

    const float row_var_sum = block_sum(local_var_sum, shared);
    const float variance = row_var_sum / static_cast<float>(width);
    const float row_rstd = rsqrtf(fmaxf(variance, 0.0f) + epsilon);

    if (threadIdx.x == 0) {
        mean[row] = row_mean;
        rstd[row] = row_rstd;
    }

    float* __restrict__ row_output = output + row * width;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float normalized = (row_input[col] - row_mean) * row_rstd;
        row_output[col] = normalized * gamma[col] + beta[col];
    }
}

}  // namespace

Status layernorm_forward(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    TensorView output,
    TensorView mean,
    TensorView rstd,
    float epsilon,
    cudaStream_t stream) {
    if (!valid_matrix_pair(input, output)) {
        return Status::failure(cudaErrorInvalidValue, "invalid layernorm input/output arguments");
    }

    const std::int64_t rows64 = input.shape.dims[0];
    const std::int64_t width64 = input.shape.dims[1];
    if (!fits_int(rows64) || !fits_int(width64) || !valid_vector(gamma, width64) || !valid_vector(beta, width64) ||
        !valid_vector(mean, rows64) || !valid_vector(rstd, rows64)) {
        return Status::failure(cudaErrorInvalidValue, "invalid layernorm shape arguments");
    }
    if (gamma.device_id != input.device_id || beta.device_id != input.device_id || mean.device_id != input.device_id ||
        rstd.device_id != input.device_id) {
        return Status::failure(cudaErrorInvalidValue, "layernorm tensors must share a device");
    }

    DeviceGuard guard(input.device_id);
    const std::size_t shared_bytes = ((kLayerNormBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    layernorm_forward_kernel<<<static_cast<unsigned int>(rows64), kLayerNormBlockSize, shared_bytes, stream>>>(
        input.data_as<const float>(),
        gamma.data_as<const float>(),
        beta.data_as<const float>(),
        output.data_as<float>(),
        mean.data_as<float>(),
        rstd.data_as<float>(),
        static_cast<int>(rows64),
        static_cast<int>(width64),
        epsilon);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
