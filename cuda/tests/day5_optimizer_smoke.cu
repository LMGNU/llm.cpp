#include "../includes/adamw.cuh"
#include "../includes/global_norm.cuh"
#include "../includes/memory.cuh"
#include "../includes/runtime.cuh"
#include "../includes/schedulers.h"
#include "../includes/tensor.cuh"

#include <cmath>
#include <cstdio>

namespace {

bool close_enough(float got, float expected, float tolerance = 1.0e-5f) {
    return std::fabs(got - expected) <= tolerance;
}

bool check_values(const char* name, const float* got, const float* expected, int count, float tolerance = 1.0e-5f) {
    for (int i = 0; i < count; ++i) {
        if (!close_enough(got[i], expected[i], tolerance)) {
            std::fprintf(stderr, "%s mismatch at %d: got %f expected %f\n", name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

void adamw_cpu(
    float* params,
    const float* grads,
    float* m,
    float* v,
    int n,
    const quadtrix::cuda::AdamWConfig& config,
    float grad_scale) {
    const float beta1_correction = 1.0f - std::pow(config.beta1, static_cast<float>(config.step));
    const float beta2_correction = 1.0f - std::pow(config.beta2, static_cast<float>(config.step));
    for (int i = 0; i < n; ++i) {
        const float param = params[i];
        const float grad = grads[i] * grad_scale;
        m[i] = config.beta1 * m[i] + (1.0f - config.beta1) * grad;
        v[i] = config.beta2 * v[i] + (1.0f - config.beta2) * grad * grad;
        const float m_hat = m[i] / beta1_correction;
        const float v_hat = v[i] / beta2_correction;
        params[i] = param - config.learning_rate *
                                (m_hat / (std::sqrt(v_hat) + config.epsilon) + config.weight_decay * param);
    }
}

}  // namespace

int main() {
    using namespace quadtrix::cuda;

    if (device_count() <= 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    constexpr int n = 8;
    const std::int64_t dims[] = {n};
    const std::int64_t partial_dims[] = {4};

    Tensor params(dims, 1, DType::F32, 0);
    Tensor grads(dims, 1, DType::F32, 0);
    Tensor m(dims, 1, DType::F32, 0);
    Tensor v(dims, 1, DType::F32, 0);
    Tensor partial(partial_dims, 1, DType::F32, 0);

    float host_params[n] = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, -0.25f, 1.5f, -1.25f};
    float host_grads[n] = {0.1f, -0.2f, 0.05f, -0.4f, 0.3f, -0.1f, 0.2f, -0.05f};
    float host_m[n] = {};
    float host_v[n] = {};

    float expected_params[n] = {};
    float expected_m[n] = {};
    float expected_v[n] = {};
    for (int i = 0; i < n; ++i) {
        expected_params[i] = host_params[i];
    }

    AdamWConfig config;
    config.learning_rate = 1.0e-3f;
    config.beta1 = 0.9f;
    config.beta2 = 0.95f;
    config.epsilon = 1.0e-8f;
    config.weight_decay = 0.01f;
    config.step = 1;

    adamw_cpu(expected_params, host_grads, expected_m, expected_v, n, config, 1.0f);

    Stream stream = Stream::create();
    if (!copy_h2d(params.data(), host_params, sizeof(host_params), stream.handle).ok ||
        !copy_h2d(grads.data(), host_grads, sizeof(host_grads), stream.handle).ok ||
        !memset_device(m.data(), 0, m.bytes(), stream.handle).ok ||
        !memset_device(v.data(), 0, v.bytes(), stream.handle).ok) {
        return 2;
    }

    Status update = adamw_update(params.view(), grads.view(), m.view(), v.view(), config, 1.0f, stream.handle);
    if (!update.ok) {
        return 3;
    }

    float got_params[n] = {};
    float got_m[n] = {};
    float got_v[n] = {};
    if (!copy_d2h(got_params, params.data(), sizeof(got_params), stream.handle).ok ||
        !copy_d2h(got_m, m.data(), sizeof(got_m), stream.handle).ok ||
        !copy_d2h(got_v, v.data(), sizeof(got_v), stream.handle).ok) {
        return 4;
    }
    stream.synchronize();

    if (!check_values("adamw params", got_params, expected_params, n) ||
        !check_values("adamw m", got_m, expected_m, n) ||
        !check_values("adamw v", got_v, expected_v, n)) {
        return 5;
    }

    const float norm = std::sqrt(0.1f * 0.1f + 0.2f * 0.2f + 0.05f * 0.05f + 0.4f * 0.4f + 0.3f * 0.3f +
                                 0.1f * 0.1f + 0.2f * 0.2f + 0.05f * 0.05f);
    const float max_norm = 0.25f;
    const float scale = clip_scale(norm, max_norm);
    float expected_clipped[n] = {};
    for (int i = 0; i < n; ++i) {
        expected_clipped[i] = host_grads[i] * scale;
    }

    Status clip_status = clip_gradients_by_global_norm(grads.view(), norm, max_norm, stream.handle);
    if (!clip_status.ok) {
        return 6;
    }

    float got_clipped[n] = {};
    if (!copy_d2h(got_clipped, grads.data(), sizeof(got_clipped), stream.handle).ok) {
        return 7;
    }
    stream.synchronize();

    if (!check_values("clipped grads", got_clipped, expected_clipped, n)) {
        return 8;
    }

    float lr = cosine_learning_rate(4, 2, 10, 1.0f, 0.1f);
    if (!(lr < 1.0f && lr > 0.1f)) {
        return 9;
    }

    std::printf("Day 5 optimizer smoke test passed\n");
    return 0;
}
