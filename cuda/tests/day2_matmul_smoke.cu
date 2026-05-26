#include "../includes/matmul.cuh"
#include "../includes/memory.cuh"
#include "../includes/runtime.cuh"
#include "../includes/tensor.cuh"

#include <cmath>
#include <cstdio>

namespace {

bool close_enough(float got, float expected) {
    return std::fabs(got - expected) < 1.0e-4f;
}

bool check_values(const char* name, const float* got, const float* expected, int count) {
    for (int i = 0; i < count; ++i) {
        if (!close_enough(got[i], expected[i])) {
            std::fprintf(stderr, "%s mismatch at %d: got %f expected %f\n", name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace quadtrix::cuda;

    if (device_count() <= 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    const std::int64_t input_dims[] = {2, 3};
    const std::int64_t weight_dims[] = {3, 2};
    const std::int64_t output_dims[] = {2, 2};

    Tensor input(input_dims, 2, DType::F32, 0);
    Tensor weight(weight_dims, 2, DType::F32, 0);
    Tensor output(output_dims, 2, DType::F32, 0);
    Tensor grad_output(output_dims, 2, DType::F32, 0);
    Tensor grad_input(input_dims, 2, DType::F32, 0);
    Tensor grad_weight(weight_dims, 2, DType::F32, 0);

    const float host_input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float host_weight[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const float host_grad_output[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float host_output[4] = {};
    float host_grad_input[6] = {};
    float host_grad_weight[6] = {};

    const float expected_output[4] = {58.0f, 64.0f, 139.0f, 154.0f};
    const float expected_grad_input[6] = {23.0f, 29.0f, 35.0f, 53.0f, 67.0f, 81.0f};
    const float expected_grad_weight[6] = {13.0f, 18.0f, 17.0f, 24.0f, 21.0f, 30.0f};

    Stream stream = Stream::create();
    if (!copy_h2d(input.data(), host_input, sizeof(host_input), stream.handle).ok ||
        !copy_h2d(weight.data(), host_weight, sizeof(host_weight), stream.handle).ok ||
        !copy_h2d(grad_output.data(), host_grad_output, sizeof(host_grad_output), stream.handle).ok) {
        return 2;
    }

    BlasHandle blas(0);
    BlasStatus forward = matmul_forward(blas, input.view(), weight.view(), output.view(), stream.handle);
    if (!forward.ok) {
        std::fprintf(stderr, "matmul_forward failed with %s: %s\n", cublas_status_name(forward.cublas_status), forward.message);
        return 3;
    }

    BlasStatus backward_input = matmul_backward_input(blas, grad_output.view(), weight.view(), grad_input.view(), stream.handle);
    if (!backward_input.ok) {
        std::fprintf(
            stderr,
            "matmul_backward_input failed with %s: %s\n",
            cublas_status_name(backward_input.cublas_status),
            backward_input.message);
        return 4;
    }

    BlasStatus backward_weight = matmul_backward_weight(blas, input.view(), grad_output.view(), grad_weight.view(), stream.handle);
    if (!backward_weight.ok) {
        std::fprintf(
            stderr,
            "matmul_backward_weight failed with %s: %s\n",
            cublas_status_name(backward_weight.cublas_status),
            backward_weight.message);
        return 5;
    }

    if (!copy_d2h(host_output, output.data(), sizeof(host_output), stream.handle).ok ||
        !copy_d2h(host_grad_input, grad_input.data(), sizeof(host_grad_input), stream.handle).ok ||
        !copy_d2h(host_grad_weight, grad_weight.data(), sizeof(host_grad_weight), stream.handle).ok) {
        return 6;
    }
    stream.synchronize();

    if (!check_values("output", host_output, expected_output, 4) ||
        !check_values("grad_input", host_grad_input, expected_grad_input, 6) ||
        !check_values("grad_weight", host_grad_weight, expected_grad_weight, 6)) {
        return 7;
    }

    std::printf("Day 2 cuBLAS matmul smoke test passed\n");
    return 0;
}
