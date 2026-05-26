#include "../includes/attention.cuh"
#include "../includes/memory.cuh"
#include "../includes/runtime.cuh"
#include "../includes/tensor.cuh"

#include <cmath>
#include <cstdio>

namespace {

bool close_enough(float got, float expected, float tolerance = 2.0e-4f) {
    return std::fabs(got - expected) <= tolerance;
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

void attention_forward_cpu(float* out, float* preatt, float* att, const float* inp, int B, int T, int C, int NH) {
    const int C3 = 3 * C;
    const int hs = C / NH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const float* query = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_row = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_row = att + b * NH * T * T + h * T * T + t * T;
                float maxval = -INFINITY;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* key = inp + b * T * C3 + t2 * C3 + C + h * hs;
                    float score = 0.0f;
                    for (int i = 0; i < hs; ++i) {
                        score += query[i] * key[i];
                    }
                    score *= scale;
                    preatt_row[t2] = score;
                    maxval = std::fmax(maxval, score);
                }
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; ++t2) {
                    att_row[t2] = std::exp(preatt_row[t2] - maxval);
                    sum += att_row[t2];
                }
                for (int t2 = 0; t2 < T; ++t2) {
                    att_row[t2] = t2 <= t ? att_row[t2] / sum : 0.0f;
                    if (t2 > t) {
                        preatt_row[t2] = 0.0f;
                    }
                }
                float* out_row = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; ++i) {
                    out_row[i] = 0.0f;
                    for (int t2 = 0; t2 <= t; ++t2) {
                        const float* value = inp + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                        out_row[i] += att_row[t2] * value[i];
                    }
                }
            }
        }
    }
}

void attention_backward_cpu(
    float* dinp,
    float* dpreatt,
    float* datt,
    const float* dout,
    const float* inp,
    const float* att,
    int B,
    int T,
    int C,
    int NH) {
    const int C3 = 3 * C;
    const int hs = C / NH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const float* att_row = att + b * NH * T * T + h * T * T + t * T;
                float* datt_row = datt + b * NH * T * T + h * T * T + t * T;
                float* dpreatt_row = dpreatt + b * NH * T * T + h * T * T + t * T;
                const float* query = inp + b * T * C3 + t * C3 + h * hs;
                float* dquery = dinp + b * T * C3 + t * C3 + h * hs;
                const float* dout_row = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* value = inp + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                    float* dvalue = dinp + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                    for (int i = 0; i < hs; ++i) {
                        datt_row[t2] += value[i] * dout_row[i];
                        dvalue[i] += att_row[t2] * dout_row[i];
                    }
                }
                for (int t2 = 0; t2 <= t; ++t2) {
                    for (int t3 = 0; t3 <= t; ++t3) {
                        const float indicator = t2 == t3 ? 1.0f : 0.0f;
                        dpreatt_row[t3] += att_row[t2] * (indicator - att_row[t3]) * datt_row[t2];
                    }
                }
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* key = inp + b * T * C3 + t2 * C3 + C + h * hs;
                    float* dkey = dinp + b * T * C3 + t2 * C3 + C + h * hs;
                    const float dp = dpreatt_row[t2] * scale;
                    for (int i = 0; i < hs; ++i) {
                        dquery[i] += key[i] * dp;
                        dkey[i] += query[i] * dp;
                    }
                }
            }
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

    constexpr int B = 1;
    constexpr int T = 3;
    constexpr int C = 4;
    constexpr int NH = 2;
    constexpr int C3 = 3 * C;
    constexpr int qkv_count = B * T * C3;
    constexpr int out_count = B * T * C;
    constexpr int att_count = B * NH * T * T;

    const std::int64_t qkv_dims[] = {B, T, C3};
    const std::int64_t out_dims[] = {B, T, C};
    const std::int64_t att_dims[] = {B, NH, T, T};

    Tensor input(qkv_dims, 3, DType::F32, 0);
    Tensor preatt(att_dims, 4, DType::F32, 0);
    Tensor att(att_dims, 4, DType::F32, 0);
    Tensor output(out_dims, 3, DType::F32, 0);
    Tensor grad_output(out_dims, 3, DType::F32, 0);
    Tensor grad_input(qkv_dims, 3, DType::F32, 0);
    Tensor grad_preatt(att_dims, 4, DType::F32, 0);
    Tensor grad_att(att_dims, 4, DType::F32, 0);

    float host_input[qkv_count] = {};
    float host_grad_output[out_count] = {};
    for (int i = 0; i < qkv_count; ++i) {
        host_input[i] = 0.01f * static_cast<float>((i % 17) - 8);
    }
    for (int i = 0; i < out_count; ++i) {
        host_grad_output[i] = 0.02f * static_cast<float>((i % 11) - 5);
    }

    float expected_output[out_count] = {};
    float expected_preatt[att_count] = {};
    float expected_att[att_count] = {};
    float expected_grad_input[qkv_count] = {};
    float expected_grad_preatt[att_count] = {};
    float expected_grad_att[att_count] = {};
    attention_forward_cpu(expected_output, expected_preatt, expected_att, host_input, B, T, C, NH);
    attention_backward_cpu(
        expected_grad_input,
        expected_grad_preatt,
        expected_grad_att,
        host_grad_output,
        host_input,
        expected_att,
        B,
        T,
        C,
        NH);

    Stream stream = Stream::create();
    if (!copy_h2d(input.data(), host_input, sizeof(host_input), stream.handle).ok ||
        !copy_h2d(grad_output.data(), host_grad_output, sizeof(host_grad_output), stream.handle).ok ||
        !memset_device(grad_input.data(), 0, grad_input.bytes(), stream.handle).ok ||
        !memset_device(grad_preatt.data(), 0, grad_preatt.bytes(), stream.handle).ok ||
        !memset_device(grad_att.data(), 0, grad_att.bytes(), stream.handle).ok) {
        return 2;
    }

    Status forward = attention_forward(input.view(), preatt.view(), att.view(), output.view(), NH, stream.handle);
    if (!forward.ok) {
        return 3;
    }
    Status backward = attention_backward(
        grad_output.view(),
        input.view(),
        att.view(),
        grad_input.view(),
        grad_preatt.view(),
        grad_att.view(),
        NH,
        stream.handle);
    if (!backward.ok) {
        return 4;
    }

    float host_output[out_count] = {};
    float host_preatt[att_count] = {};
    float host_att[att_count] = {};
    float host_grad_input[qkv_count] = {};
    float host_grad_preatt[att_count] = {};
    float host_grad_att[att_count] = {};
    if (!copy_d2h(host_output, output.data(), sizeof(host_output), stream.handle).ok ||
        !copy_d2h(host_preatt, preatt.data(), sizeof(host_preatt), stream.handle).ok ||
        !copy_d2h(host_att, att.data(), sizeof(host_att), stream.handle).ok ||
        !copy_d2h(host_grad_input, grad_input.data(), sizeof(host_grad_input), stream.handle).ok ||
        !copy_d2h(host_grad_preatt, grad_preatt.data(), sizeof(host_grad_preatt), stream.handle).ok ||
        !copy_d2h(host_grad_att, grad_att.data(), sizeof(host_grad_att), stream.handle).ok) {
        return 5;
    }
    stream.synchronize();

    if (!check_values("output", host_output, expected_output, out_count) ||
        !check_values("preatt", host_preatt, expected_preatt, att_count) ||
        !check_values("att", host_att, expected_att, att_count) ||
        !check_values("grad_input", host_grad_input, expected_grad_input, qkv_count) ||
        !check_values("grad_preatt", host_grad_preatt, expected_grad_preatt, att_count) ||
        !check_values("grad_att", host_grad_att, expected_grad_att, att_count)) {
        return 6;
    }

    std::printf("Day 4 attention smoke test passed\n");
    return 0;
}
