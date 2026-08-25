// On-device check of gqa_kv_hadamard64 in BOTH one-warp and multi-warp
// launches. The helper is transcribed verbatim from the production
// gqa_attention_kv_quant.cuh (including the lane-parity fix). The multi-warp
// case (four warps per CTA, data lane = threadIdx.x & 31) is the exact
// geometry of the fused GQA kernels; the pre-fix helper used the absolute
// threadIdx.x in its parity selector, corrupting every rotated KV in any CTA
// with more than one warp.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kLanes = 32;
constexpr int kDims  = 64;
constexpr int kTests = 4;

// Verbatim from gqa_attention_kv_quant.cuh (with the lane fix).
__device__ __forceinline__ void gqa_kv_hadamard64(float& x0, float& x1,
                                                  unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        // Butterfly parity is in LANE space (data lane = threadIdx.x & 31).
        const bool hi = ((static_cast<int>(threadIdx.x) & 31) & offset) != 0;
        x0            = hi ? y0 - x0 : x0 + y0;
        x1            = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

// One 64-dim vector per warp: lane l holds elements l and l+32.
__global__ void rotate_twice_kernel(const float* in, float* out) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int base    = warp_id * kDims;
    const int l       = static_cast<int>(threadIdx.x) & 31;
    const int i0      = base + l;
    const int i1      = base + l + 32;
    float x0          = in[i0];
    float x1          = in[i1];
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
    cudaMalloc(&d_in, in.size() * sizeof(float));
    cudaMalloc(&d_out, in.size() * sizeof(float));
    cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);

    int failures = 0;
    // One warp per CTA (the standalone geometry) and four warps per CTA (the
    // fused-kernel geometry). Both must round-trip to identity.
    for (const int warps_per_cta : {1, 4}) {
        const int blocks = kTests / warps_per_cta;
        rotate_twice_kernel<<<blocks, warps_per_cta * kLanes>>>(d_in, d_out);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            std::printf("launch error warps=%d\n", warps_per_cta);
            ++failures;
            continue;
        }
        std::vector<float> out(in.size());
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int i = 0; i < static_cast<int>(in.size()); ++i) {
            const double e = std::fabs(static_cast<double>(out[i]) - in[i]);
            if (e > max_err) max_err = e;
        }
        const bool ok = max_err < 1e-3;
        std::printf("rotate-twice warps_per_cta=%d max_abs_err=%.3e %s\n", warps_per_cta,
                    max_err, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return failures == 0 ? 0 : 1;
}