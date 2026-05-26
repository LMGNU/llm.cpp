#include "../includes/attention.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>
#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

bool valid_f32_cuda(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.is_contiguous();
}

__global__ void attention_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ att,
    float* __restrict__ grad_input,
    float* __restrict__ grad_preatt,
    float* __restrict__ grad_att,
    int total_rows,
    int time,
    int channels,
    int num_heads,
    int head_size) {
    const int row_id = blockIdx.x;
    if (row_id >= total_rows) {
        return;
    }

    const int h = row_id % num_heads;
    const int t = (row_id / num_heads) % time;
    const int b = row_id / (num_heads * time);
    const int c3 = 3 * channels;
    const float scale = rsqrtf(static_cast<float>(head_size));
    const float* __restrict__ query = input + b * time * c3 + t * c3 + h * head_size;
    float* __restrict__ dquery = grad_input + b * time * c3 + t * c3 + h * head_size;
    const float* __restrict__ dout = grad_output + b * time * channels + t * channels + h * head_size;
    const float* __restrict__ att_row = att + b * num_heads * time * time + h * time * time + t * time;
    float* __restrict__ datt_row = grad_att + b * num_heads * time * time + h * time * time + t * time;
    float* __restrict__ dpreatt_row = grad_preatt + b * num_heads * time * time + h * time * time + t * time;

    for (int t2 = threadIdx.x; t2 <= t; t2 += blockDim.x) {
        const float* __restrict__ value = input + b * time * c3 + t2 * c3 + 2 * channels + h * head_size;
        float* __restrict__ dvalue = grad_input + b * time * c3 + t2 * c3 + 2 * channels + h * head_size;
        float local_datt = 0.0f;
        for (int i = 0; i < head_size; ++i) {
            local_datt += value[i] * dout[i];
            atomicAdd(dvalue + i, att_row[t2] * dout[i]);
        }
        datt_row[t2] += local_datt;
    }
    __syncthreads();

    for (int t3 = threadIdx.x; t3 <= t; t3 += blockDim.x) {
        float dp = 0.0f;
        const float att_t3 = att_row[t3];
        for (int t2 = 0; t2 <= t; ++t2) {
            const float indicator = t2 == t3 ? 1.0f : 0.0f;
            dp += att_row[t2] * (indicator - att_t3) * datt_row[t2];
        }
        dpreatt_row[t3] += dp;
    }
    __syncthreads();

    for (int t2 = threadIdx.x; t2 <= t; t2 += blockDim.x) {
        const float* __restrict__ key = input + b * time * c3 + t2 * c3 + channels + h * head_size;
        float* __restrict__ dkey = grad_input + b * time * c3 + t2 * c3 + channels + h * head_size;
        const float dp = dpreatt_row[t2] * scale;
        for (int i = 0; i < head_size; ++i) {
            atomicAdd(dquery + i, key[i] * dp);
            atomicAdd(dkey + i, query[i] * dp);
        }
    }
}

}  // namespace

Status attention_backward(
    const TensorView& grad_output,
    const TensorView& input_qkv,
    const TensorView& att,
    TensorView grad_input_qkv,
    TensorView grad_preatt,
    TensorView grad_att,
    int num_heads,
    cudaStream_t stream) {
    if (!valid_f32_cuda(grad_output) || !valid_f32_cuda(input_qkv) || !valid_f32_cuda(att) ||
        !valid_f32_cuda(grad_input_qkv) || !valid_f32_cuda(grad_preatt) || !valid_f32_cuda(grad_att) ||
        input_qkv.shape.rank != 3 || grad_output.shape.rank != 3 || att.shape.rank != 4 ||
        grad_input_qkv.shape.rank != 3 || grad_preatt.shape.rank != 4 || grad_att.shape.rank != 4) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_backward tensors");
    }

    const std::int64_t batch = input_qkv.shape.dims[0];
    const std::int64_t time = input_qkv.shape.dims[1];
    const std::int64_t channels3 = input_qkv.shape.dims[2];
    if (num_heads <= 0 || channels3 % 3 != 0) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_backward qkv shape");
    }
    const std::int64_t channels = channels3 / 3;
    if (channels % num_heads != 0 || grad_output.shape.dims[0] != batch || grad_output.shape.dims[1] != time ||
        grad_output.shape.dims[2] != channels || grad_input_qkv.shape.dims != input_qkv.shape.dims ||
        att.shape.dims[0] != batch || att.shape.dims[1] != num_heads || att.shape.dims[2] != time ||
        att.shape.dims[3] != time || grad_preatt.shape.dims != att.shape.dims ||
        grad_att.shape.dims != att.shape.dims || grad_output.device_id != input_qkv.device_id ||
        att.device_id != input_qkv.device_id || grad_input_qkv.device_id != input_qkv.device_id ||
        grad_preatt.device_id != input_qkv.device_id || grad_att.device_id != input_qkv.device_id ||
        !fits_int(batch) || !fits_int(time) || !fits_int(channels)) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_backward shape");
    }

    DeviceGuard guard(input_qkv.device_id);
    const int rows = static_cast<int>(batch * time * num_heads);
    attention_backward_kernel<<<rows, kDefaultBlockSize, 0, stream>>>(
        grad_output.data_as<const float>(),
        input_qkv.data_as<const float>(),
        att.data_as<const float>(),
        grad_input_qkv.data_as<float>(),
        grad_preatt.data_as<float>(),
        grad_att.data_as<float>(),
        rows,
        static_cast<int>(time),
        static_cast<int>(channels),
        num_heads,
        static_cast<int>(channels / num_heads));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
