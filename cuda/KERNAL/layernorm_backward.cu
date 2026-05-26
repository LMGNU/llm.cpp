#include "../includes/layernorm.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

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

bool valid_matrix(const TensorView& tensor, std::int64_t rows, std::int64_t width) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.rank == 2 && tensor.shape.dims[0] == rows && tensor.shape.dims[1] == width &&
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

__global__ void layernorm_backward_input_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_input,
    int rows,
    int width) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float row_mean = mean[row];
    const float row_rstd = rstd[row];
    const float* row_grad_output = grad_output + row * width;
    const float* row_input = input + row * width;

    float local_sum_dy_gamma = 0.0f;
    float local_sum_dy_gamma_xhat = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float xhat = (row_input[col] - row_mean) * row_rstd;
        const float dy_gamma = row_grad_output[col] * gamma[col];
        local_sum_dy_gamma += dy_gamma;
        local_sum_dy_gamma_xhat += dy_gamma * xhat;
    }

    const float inv_width = 1.0f / static_cast<float>(width);
    const float row_sum_dy_gamma = block_sum(local_sum_dy_gamma, shared);
    const float row_sum_dy_gamma_xhat = block_sum(local_sum_dy_gamma_xhat, shared);
    float* __restrict__ row_grad_input = grad_input + row * width;

    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float xhat = (row_input[col] - row_mean) * row_rstd;
        const float dy = row_grad_output[col];
        const float dy_gamma = dy * gamma[col];
        row_grad_input[col] +=
            row_rstd * (dy_gamma - inv_width * row_sum_dy_gamma - xhat * inv_width * row_sum_dy_gamma_xhat);
    }
}

__global__ void layernorm_backward_param_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows,
    int width) {
    extern __shared__ float shared[];

    const int col = blockIdx.x;
    if (col >= width) {
        return;
    }

    float local_gamma = 0.0f;
    float local_beta = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const int idx = row * width + col;
        const float dy = grad_output[idx];
        const float xhat = (input[idx] - mean[row]) * rstd[row];
        local_gamma += dy * xhat;
        local_beta += dy;
    }

    const float gamma_sum = block_sum(local_gamma, shared);
    const float beta_sum = block_sum(local_beta, shared);

    if (threadIdx.x == 0) {
        grad_gamma[col] += gamma_sum;
        grad_beta[col] += beta_sum;
    }
}

}  // namespace

Status layernorm_backward(
    const TensorView& grad_output,
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& mean,
    const TensorView& rstd,
    TensorView grad_input,
    TensorView grad_gamma,
    TensorView grad_beta,
    cudaStream_t stream) {
    if (input.shape.rank != 2 || !fits_int(input.shape.dims[0]) || !fits_int(input.shape.dims[1])) {
        return Status::failure(cudaErrorInvalidValue, "invalid layernorm_backward input shape");
    }

    const std::int64_t rows64 = input.shape.dims[0];
    const std::int64_t width64 = input.shape.dims[1];
    if (!valid_matrix(input, rows64, width64) || !valid_matrix(grad_output, rows64, width64) ||
        !valid_matrix(grad_input, rows64, width64) || !valid_vector(gamma, width64) ||
        !valid_vector(mean, rows64) || !valid_vector(rstd, rows64) || !valid_vector(grad_gamma, width64) ||
        !valid_vector(grad_beta, width64)) {
        return Status::failure(cudaErrorInvalidValue, "invalid layernorm_backward tensor arguments");
    }
    if (grad_output.device_id != input.device_id || gamma.device_id != input.device_id ||
        mean.device_id != input.device_id || rstd.device_id != input.device_id ||
        grad_input.device_id != input.device_id || grad_gamma.device_id != input.device_id ||
        grad_beta.device_id != input.device_id) {
        return Status::failure(cudaErrorInvalidValue, "layernorm_backward tensors must share a device");
    }

    DeviceGuard guard(input.device_id);

    const std::size_t shared_bytes = ((kLayerNormBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    layernorm_backward_input_kernel<<<static_cast<unsigned int>(rows64), kLayerNormBlockSize, shared_bytes, stream>>>(
        grad_output.data_as<const float>(),
        input.data_as<const float>(),
        gamma.data_as<const float>(),
        mean.data_as<const float>(),
        rstd.data_as<const float>(),
        grad_input.data_as<float>(),
        static_cast<int>(rows64),
        static_cast<int>(width64));
    Status input_status = QUADTRIX_CUDA_CHECK(cudaGetLastError());
    if (!input_status.ok) {
        return input_status;
    }

    layernorm_backward_param_kernel<<<static_cast<unsigned int>(width64), kLayerNormBlockSize, shared_bytes, stream>>>(
        grad_output.data_as<const float>(),
        input.data_as<const float>(),
        mean.data_as<const float>(),
        rstd.data_as<const float>(),
        grad_gamma.data_as<float>(),
        grad_beta.data_as<float>(),
        static_cast<int>(rows64),
        static_cast<int>(width64));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
