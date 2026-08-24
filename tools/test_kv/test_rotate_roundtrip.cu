// Minimal on-device check of gqa_kv_hadamard64: is the 64-wide rotate
// round trip identity on sm_120a? This is the one primitive unique to the
// rk* compressed-KV modes (which produce garbage in the model while
// bf16/int8 are clean). The transform body is transcribed verbatim from
// src/ops/kernel/gqa_attention_kv_quant.cuh so the device test exercises
// the exact production arithmetic.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kLanes = 32;
constexpr int kDims  = 64;
constexpr int kTests = 4;

// Verbatim from gqa_attention_kv_quant.cuh:80-95.
__device__ __forceinline__ void gqa_kv_hadamard64(float& x0, float& x1,
                                                  unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool hi  = (static_cast<int>(threadIdx.x) & offset) != 0;
        x0             = hi ? y0 - x0 : x0 + y0;
        x1             = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

// Each block is one 64-dim vector: lane l holds elements l and l+32.
__global__ void rotate_twice_kernel(const float* in, float* out) {
    const int base = blockIdx.x * kDims;
    const int l    = threadIdx.x;
    const int i0   = base + l;
    const int i1   = base + l + 32;
    float x0       = in[i0];
    float x1       = in[i1];
    gqa_kv_hadamard64(x0, x1);
    gqa_kv_hadamard64(x0, x1);
    out[i0] = x0;
    out[i1] = x1;
}

} // namespace

int main() {
    std::vector<float> in(kDims * kTests);
    for (int t = 0; t < kTests; ++t) {
        const float scale = 1.0f + static_cast<float>(t);
        for (int i = 0; i < kDims; ++i) {
            in[t * kDims + i] = scale * ((i % 7) - 3.0f) + 0.25f * (i % 3);
        }
    }

    float *d_in = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_in, in.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, in.size() * sizeof(float)) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);

    rotate_twice_kernel<<<kTests, kLanes>>>(d_in, d_out);
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out(in.size());
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    double max_err = 0.0;
    for (int i = 0; i < static_cast<int>(in.size()); ++i) {
        const double e = std::fabs(static_cast<double>(out[i]) - in[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("rotate-twice max_abs_err=%.3e %s\n", max_err,
                max_err < 1e-3 ? "OK" : "FAIL");
    return max_err < 1e-3 ? 0 : 1;
}