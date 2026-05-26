#pragma once

#include "tensor.cuh"

#include <cstddef>

namespace quadtrix {
namespace cuda {

struct ShardRange {
    std::size_t offset = 0;
    std::size_t length = 0;
};

inline ShardRange zero_shard_range(std::size_t total_elements, int world_size, int rank) {
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        return {};
    }

    const std::size_t base = total_elements / static_cast<std::size_t>(world_size);
    const std::size_t rem = total_elements % static_cast<std::size_t>(world_size);
    const std::size_t rank_size = base + (static_cast<std::size_t>(rank) < rem ? 1 : 0);
    const std::size_t rank_offset =
        base * static_cast<std::size_t>(rank) +
        (static_cast<std::size_t>(rank) < rem ? static_cast<std::size_t>(rank) : rem);
    return {rank_offset, rank_size};
}

inline TensorView zero_shard_view(TensorView tensor, int world_size, int rank) {
    ShardRange range = zero_shard_range(tensor.numel(), world_size, rank);
    TensorView shard = tensor;
    shard.data = static_cast<char*>(tensor.data) + range.offset * dtype_size(tensor.dtype);
    if (range.length == 0) {
        shard.shape = TensorShape{};
        return shard;
    }
    const std::int64_t shard_length = static_cast<std::int64_t>(range.length);
    shard.shape = TensorShape::contiguous(&shard_length, 1);
    return shard;
}

}  // namespace cuda
}  // namespace quadtrix
