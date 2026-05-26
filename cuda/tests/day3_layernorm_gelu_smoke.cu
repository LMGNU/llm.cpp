#include "../includes/gelu.cuh"
#include "../includes/layernorm.cuh"
#include "../includes/memory.cuh"
#include "../includes/runtime.cuh"
#include "../includes/tensor.cuh"

#include <cmath>
#include <cstdio>

namespace {

constexpr float kSqrtTwoOverPi = 0.79788456080286535588f;
constexpr float kGeluCoeff = 0.044715f;

float gelu_approx(float x) {
    return 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi * (x + kGeluCoeff * x * x * x)));
}

float gelu_approx_grad(float x) {
    const float x2 = x * x;
    const float inner = kSqrtTwoOverPi * (x + kGeluCoeff * x2 * x);
    const float t = std::tanh(inner);
    const float sech2 = 1.0f - t * t;
    const float inner_grad = kSqrtTwoOverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_grad;
}

bool close_enough(float got, float expected, float tolerance = 1.0e-4f) {
    return std::fabs(got - expected) <= tolerance;
}

bool check_values(const char* name, const float* got, const float* expected, int count, float tolerance = 1.0e-4f) {
    for (int i = 0; i < count; ++i) {
        if (!close_enough(got[i], expected[i], tolerance)) {
            std::fprintf(stderr, "%s mismatch at %d: got %f expected %f\n", name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

void layernorm_forward_cpu(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* rstd,
    int rows,
    int width,
    float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const float* row_input = input + row * width;
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int col = 0; col < width; ++col) {
            sum += row_input[col];
            sq_sum += row_input[col] * row_input[col];
        }

        mean[row] = sum / static_cast<float>(width);
        const float variance = sq_sum / static_cast<float>(width) - mean[row] * mean[row];
        rstd[row] = 1.0f / std::sqrt(variance + epsilon);
        for (int col = 0; col < width; ++col) {
            const float xhat = (row_input[col] - mean[row]) * rstd[row];
            output[row * width + col] = xhat * gamma[col] + beta[col];
        }
    }
}

void layernorm_backward_cpu(
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* mean,
    const float* rstd,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int width) {
    for (int col = 0; col < width; ++col) {
        grad_gamma[col] = 0.0f;
        grad_beta[col] = 0.0f;
    }

    for (int row = 0; row < rows; ++row) {
        float sum_dy_gamma = 0.0f;
        float sum_dy_gamma_xhat = 0.0f;
        for (int col = 0; col < width; ++col) {
            const int idx = row * width + col;
            const float xhat = (input[idx] - mean[row]) * rstd[row];
            const float dy_gamma = grad_output[idx] * gamma[col];
            sum_dy_gamma += dy_gamma;
            sum_dy_gamma_xhat += dy_gamma * xhat;
        }

        for (int col = 0; col < width; ++col) {
            const int idx = row * width + col;
            const float xhat = (input[idx] - mean[row]) * rstd[row];
            const float dy = grad_output[idx];
            const float dy_gamma = dy * gamma[col];
            grad_input[idx] = rstd[row] *
                              (dy_gamma - sum_dy_gamma / static_cast<float>(width) -
                               xhat * sum_dy_gamma_xhat / static_cast<float>(width));
            grad_gamma[col] += dy * xhat;
            grad_beta[col] += dy;
        }
    }
}

}  // namespace

int main() {
    using namespace quadtrix::cuda;

    if (device_count() <= 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    constexpr int rows = 2;
    constexpr int width = 4;
    constexpr int count = rows * width;
    constexpr float epsilon = 1.0e-5f;

    const std::int64_t matrix_dims[] = {rows, width};
    const std::int64_t width_dims[] = {width};
    const std::int64_t row_dims[] = {rows};

    Tensor input(matrix_dims, 2, DType::F32, 0);
    Tensor gamma(width_dims, 1, DType::F32, 0);
    Tensor beta(width_dims, 1, DType::F32, 0);
    Tensor output(matrix_dims, 2, DType::F32, 0);
    Tensor mean(row_dims, 1, DType::F32, 0);
    Tensor rstd(row_dims, 1, DType::F32, 0);
    Tensor grad_output(matrix_dims, 2, DType::F32, 0);
    Tensor grad_input(matrix_dims, 2, DType::F32, 0);
    Tensor grad_gamma(width_dims, 1, DType::F32, 0);
    Tensor grad_beta(width_dims, 1, DType::F32, 0);
    Tensor gelu_output(matrix_dims, 2, DType::F32, 0);
    Tensor gelu_grad_input(matrix_dims, 2, DType::F32, 0);

    const float host_input[count] = {-1.0f, 0.0f, 1.0f, 2.0f, 0.5f, -0.5f, 1.5f, -1.5f};
    const float host_gamma[width] = {1.0f, 1.5f, -1.0f, 0.5f};
    const float host_beta[width] = {0.1f, -0.2f, 0.3f, 0.0f};
    const float host_grad_output[count] = {0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f};

    float expected_output[count] = {};
    float expected_mean[rows] = {};
    float expected_rstd[rows] = {};
    float expected_grad_input[count] = {};
    float expected_grad_gamma[width] = {};
    float expected_grad_beta[width] = {};
    float expected_gelu_output[count] = {};
    float expected_gelu_grad_input[count] = {};

    layernorm_forward_cpu(
        host_input,
        host_gamma,
        host_beta,
        expected_output,
        expected_mean,
        expected_rstd,
        rows,
        width,
        epsilon);
    layernorm_backward_cpu(
        host_grad_output,
        host_input,
        host_gamma,
        expected_mean,
        expected_rstd,
        expected_grad_input,
        expected_grad_gamma,
        expected_grad_beta,
        rows,
        width);
    for (int i = 0; i < count; ++i) {
        expected_gelu_output[i] = gelu_approx(host_input[i]);
        expected_gelu_grad_input[i] = host_grad_output[i] * gelu_approx_grad(host_input[i]);
    }

    Stream stream = Stream::create();
    if (!copy_h2d(input.data(), host_input, sizeof(host_input), stream.handle).ok ||
        !copy_h2d(gamma.data(), host_gamma, sizeof(host_gamma), stream.handle).ok ||
        !copy_h2d(beta.data(), host_beta, sizeof(host_beta), stream.handle).ok ||
        !copy_h2d(grad_output.data(), host_grad_output, sizeof(host_grad_output), stream.handle).ok) {
        return 2;
    }
    if (!memset_device(grad_input.data(), 0, grad_input.bytes(), stream.handle).ok ||
        !memset_device(grad_gamma.data(), 0, grad_gamma.bytes(), stream.handle).ok ||
        !memset_device(grad_beta.data(), 0, grad_beta.bytes(), stream.handle).ok ||
        !memset_device(gelu_grad_input.data(), 0, gelu_grad_input.bytes(), stream.handle).ok) {
        return 3;
    }

    Status ln_forward = layernorm_forward(
        input.view(),
        gamma.view(),
        beta.view(),
        output.view(),
        mean.view(),
        rstd.view(),
        epsilon,
        stream.handle);
    if (!ln_forward.ok) {
        return 4;
    }

    Status ln_backward = layernorm_backward(
        grad_output.view(),
        input.view(),
        gamma.view(),
        mean.view(),
        rstd.view(),
        grad_input.view(),
        grad_gamma.view(),
        grad_beta.view(),
        stream.handle);
    if (!ln_backward.ok) {
        return 5;
    }

    Status gelu_fwd = gelu_forward(input.view(), gelu_output.view(), GeluMode::Approximate, stream.handle);
    if (!gelu_fwd.ok) {
        return 6;
    }

    Status gelu_bwd =
        gelu_backward(grad_output.view(), input.view(), gelu_grad_input.view(), GeluMode::Approximate, stream.handle);
    if (!gelu_bwd.ok) {
        return 7;
    }

    float host_output[count] = {};
    float host_mean[rows] = {};
    float host_rstd[rows] = {};
    float host_grad_input[count] = {};
    float host_grad_gamma[width] = {};
    float host_grad_beta[width] = {};
    float host_gelu_output[count] = {};
    float host_gelu_grad_input[count] = {};

    if (!copy_d2h(host_output, output.data(), sizeof(host_output), stream.handle).ok ||
        !copy_d2h(host_mean, mean.data(), sizeof(host_mean), stream.handle).ok ||
        !copy_d2h(host_rstd, rstd.data(), sizeof(host_rstd), stream.handle).ok ||
        !copy_d2h(host_grad_input, grad_input.data(), sizeof(host_grad_input), stream.handle).ok ||
        !copy_d2h(host_grad_gamma, grad_gamma.data(), sizeof(host_grad_gamma), stream.handle).ok ||
        !copy_d2h(host_grad_beta, grad_beta.data(), sizeof(host_grad_beta), stream.handle).ok ||
        !copy_d2h(host_gelu_output, gelu_output.data(), sizeof(host_gelu_output), stream.handle).ok ||
        !copy_d2h(host_gelu_grad_input, gelu_grad_input.data(), sizeof(host_gelu_grad_input), stream.handle).ok) {
        return 8;
    }
    stream.synchronize();

    if (!check_values("layernorm output", host_output, expected_output, count) ||
        !check_values("layernorm mean", host_mean, expected_mean, rows) ||
        !check_values("layernorm rstd", host_rstd, expected_rstd, rows) ||
        !check_values("layernorm grad_input", host_grad_input, expected_grad_input, count, 2.0e-4f) ||
        !check_values("layernorm grad_gamma", host_grad_gamma, expected_grad_gamma, width, 2.0e-4f) ||
        !check_values("layernorm grad_beta", host_grad_beta, expected_grad_beta, width) ||
        !check_values("gelu output", host_gelu_output, expected_gelu_output, count) ||
        !check_values("gelu grad_input", host_gelu_grad_input, expected_gelu_grad_input, count)) {
        return 9;
    }

    std::printf("Day 3 LayerNorm + GELU smoke test passed\n");
    return 0;
}
