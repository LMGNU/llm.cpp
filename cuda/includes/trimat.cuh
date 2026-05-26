#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status causal_mask_forward(
    TensorView matrix,
    float masked_value = 0.0f,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
