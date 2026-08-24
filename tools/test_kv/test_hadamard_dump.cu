// Dump the hadamard64 round trip element-by-element to expose the structure
// of the failure (which offsets/dims deviate, by what magnitude).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

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

__global__ void dump_kernel(const float* in, float* out1, float* out2) {
    const int l  = threadIdx.x;
    float x0     = in[l];
    float x1     = in[32 + l];
    gqa_kv_hadamard64(x0, x1);
    out1[l] = x0;
    out1[32 + l] = x1;
    gqa_kv_hadamard64(x0, x1);
    out2[l] = x0;
    out2[32 + l] = x1;
}

int main() {
    constexpr int N = 64;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i) in[i] = float(i % 8) + 0.5f * (i / 8);

    float *di, *do1, *do2;
    cudaMalloc(&di, N * sizeof(float));
    cudaMalloc(&do1, N * sizeof(float));
    cudaMalloc(&do2, N * sizeof(float));
    cudaMemcpy(di, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    dump_kernel<<<1, 32>>>(di, do1, do2);
    cudaDeviceSynchronize();
    std::vector<float> o1(N), o2(N);
    cudaMemcpy(o1.data(), do1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(o2.data(), do2, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::printf(" i    in   after1   after2\n");
    for (int i = 0; i < N; ++i) {
        std::printf("%2d %6.2f %7.2f %7.2f\n", i, in[i], o1[i], o2[i]);
    }
    return 0;
}