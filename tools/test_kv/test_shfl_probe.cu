// Probe __shfl_xor_sync semantics on the current arch: each lane writes its
// own id, then a single xor-shuffle by offset 1; host checks lane i received
// lane i^1's id. This isolates whether the hardware shuffle primitive itself
// is sound before blaming the hadamard butterfly that uses it.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void shfl_probe_kernel(int* __restrict__ out, int* __restrict__ xor1,
                                  int* __restrict__ xor32) {
    const int l = threadIdx.x;
    out[l]      = l;
    xor1[l]     = __shfl_xor_sync(0xffffffffu, l, 1);
    xor32[l]    = __shfl_xor_sync(0xffffffffu, l, 32);
}
int main() {
    constexpr int kN = 32;
    int* out; int* x1; int* x32; cudaMalloc(&out, kN*sizeof(int));
    cudaMalloc(&x1, kN*sizeof(int)); cudaMalloc(&x32, kN*sizeof(int));
    shfl_probe_kernel<<<1, kN>>>(out, x1, x32);
    cudaDeviceSynchronize();
    int h_out[kN], h_x1[kN], h_x32[kN];
    cudaMemcpy(h_out, out, sizeof(h_out), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x1, x1, sizeof(h_x1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x32, x32, sizeof(h_x32), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int l = 0; l < kN; ++l) {
        const bool ok1 = h_x1[l] == (l ^ 1);
        const bool ok32 = h_x32[l] == (l ^ 32);
        if (!ok1 || !ok32) { ++bad; }
        if (l < 8 || !ok1 || !ok32) {
            printf("lane %2d: id=%2d xor1=%2d (want %2d)%s xor32=%2d (want %2d)%s\n",
                  l, h_out[l], h_x1[l], l ^ 1, ok1 ? "" : "  <-- BAD", h_x32[l], l ^ 32,
                  ok32 ? "" : "  <-- BAD");
        }
    }
    printf("bad lanes: %d/%d %s\n", bad, kN, bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 1;
}