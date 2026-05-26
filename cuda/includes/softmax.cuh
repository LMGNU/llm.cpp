#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status softmax_forward(
    const TensorView& logits,
    TensorView probs,
    int valid_cols,
    cudaStream_t stream = nullptr);

Status causal_softmax_forward(
    const TensorView& preatt,
    TensorView att,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
