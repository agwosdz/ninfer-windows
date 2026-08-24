// Full packed-V codec round trip on sm_120a: rotate, per-group amax scale,
// int4 pair pack, then decode (unpack, dequant, rotate back) and compare
// per-dimension against the original. Mirrors the fused kernel's V path
// exactly (rk4v4/rk4v4-e8/rk2v4-e8 share this packed-V format; rk8v4 too).

#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

constexpr int kHeads  = 8;
constexpr int kD      = kGqaKvQuantHeadDim;  // 256
constexpr int kGroups = kGqaKvQuantGroups;   // 4
constexpr int kPackDim = kD / 2;             // 128 bytes per head

__device__ __forceinline__ float warp_absmax2(float a, float b) {
    float m = fmaxf(fabsf(a), fabsf(b));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    return m;
}

__device__ __forceinline__ float v_inv_use(float scale) {
    return scale > 0.0f ? 1.0f / scale : 0.0f;
}

// Encode one head: per 64-group, rotate the two 32-halves, quant+pack V.
__global__ void encode_v_kernel(const __nv_bfloat16* in, std::uint8_t* packed_v,
                                __half* v_scale) {
    const int head = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* x = in + head * kD;
    std::uint8_t* pv       = packed_v + head * kPackDim;
    __half* vs             = v_scale + head * kGroups;

    for (int grp = 0; grp < kGroups; ++grp) {
        const int d0 = grp * kGqaKvQuantGroup + lane;
        const int d1 = d0 + 32;
        float vv0    = __bfloat162float(x[d0]);
        float vv1    = __bfloat162float(x[d1]);
        gqa_kv_hadamard64(vv0, vv1);
        const float vamax = warp_absmax2(vv0, vv1);
        const __half vsh  = __float2half_rn(vamax > 0.0f ? vamax / 7.0f : 0.0f);
        const float vs_f  = __half2float(vsh);
        const float vinv  = v_inv_use(vs_f);

        const float hi0 = __shfl_down_sync(0xffffffffu, vv0, 1);
        const float hi1 = __shfl_down_sync(0xffffffffu, vv1, 1);
        if ((lane & 1) == 0) {
            pv[grp * 32 + lane / 2]       = gqa_kv_pack_i4(
                gqa_kv_quant_i4_code(vv0, vinv), gqa_kv_quant_i4_code(hi0, vinv));
            pv[grp * 32 + 16 + lane / 2] = gqa_kv_pack_i4(
                gqa_kv_quant_i4_code(vv1, vinv), gqa_kv_quant_i4_code(hi1, vinv));
        }
        if (lane == 0) { vs[grp] = vsh; }
        __syncwarp();
    }
}

// Decode back: unpack each dim from the packed pair byte, dequant, rotate.
__global__ void decode_kernel(const std::uint8_t* packed_v, const __half* v_scale,
                              __nv_bfloat16* out) {
    const int head = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const std::uint8_t* pv = packed_v + head * kPackDim;
    const __half* vs       = v_scale + head * kGroups;
    __nv_bfloat16* o       = out + head * kD;

    for (int grp = 0; grp < kGroups; ++grp) {
        const int d0 = grp * kGqaKvQuantGroup + lane;
        const int d1 = d0 + 32;
        const float s = __half2float(vs[grp]);
        const int nib = lane & 1;
        float a =
            static_cast<float>(gqa_kv_unpack_i4(pv[grp * 32 + lane / 2], nib)) * s;
        float b =
            static_cast<float>(gqa_kv_unpack_i4(pv[grp * 32 + 16 + lane / 2], nib)) * s;
        gqa_kv_hadamard64(a, b);
        o[d0] = __float2bfloat16(a);
        o[d1] = __float2bfloat16(b);
        __syncwarp();
    }
}

} // namespace

int main() {
    std::printf("[1] main start\n"); std::fflush(stdout);
    // Random bf16-like values in a realistic range (KV activations ~N(0, 1-4)).
    std::vector<std::uint16_t> in_bits(kHeads * kD);
    std::mt19937_64 rng(0x5eed);
    auto f32_to_bf16 = [](float f) {
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x40u);
        const std::uint32_t lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb;
        return static_cast<std::uint16_t>(u >> 16);
    };
    auto bf16_to_f32 = [](std::uint16_t h) {
        const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    for (int i = 0; i < kHeads * kD; ++i) {
        const float v = (static_cast<float>((rng() % 2048) - 1024) / 256.0f);
        in_bits[i]    = f32_to_bf16(v);
    }

    const std::size_t nbytes_in = static_cast<std::size_t>(kHeads) * kD * sizeof(std::uint16_t);
    const std::size_t nbytes_pk = static_cast<std::size_t>(kHeads) * kPackDim;
    const std::size_t nbytes_sc = static_cast<std::size_t>(kHeads) * kGroups * sizeof(std::uint16_t);

    std::uint16_t *d_in = nullptr, *d_out = nullptr;
    std::uint8_t* d_pk = nullptr;
    std::uint16_t* d_sc = nullptr;
    if (cudaMalloc(&d_in, nbytes_in) != cudaSuccess || cudaMalloc(&d_out, nbytes_in) != cudaSuccess ||
        cudaMalloc(&d_pk, nbytes_pk) != cudaSuccess || cudaMalloc(&d_sc, nbytes_sc) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        return 1;
    }
    std::printf("[2] buffers allocated+filled\n"); std::fflush(stdout);
    cudaMemcpy(d_in, in_bits.data(), nbytes_in, cudaMemcpyHostToDevice);
    std::printf("[3] copied to device\n"); std::fflush(stdout);

    encode_v_kernel<<<kHeads, 32>>>(reinterpret_cast<const __nv_bfloat16*>(d_in),
                                    reinterpret_cast<std::uint8_t*>(d_pk),
                                    reinterpret_cast<__half*>(d_sc));
    std::printf("[4] encode launched\n"); std::fflush(stdout);
    decode_kernel<<<kHeads, 32>>>(d_pk, reinterpret_cast<const __half*>(d_sc),
                                  reinterpret_cast<__nv_bfloat16*>(d_out));
    std::printf("[5] decode launched\n"); std::fflush(stdout);
    cudaDeviceSynchronize();
    std::printf("[6] synced\n"); std::fflush(stdout);
    const cudaError_t lerr = cudaGetLastError();
    if (lerr != cudaSuccess) {
        std::printf("launch error: %s\n", cudaGetErrorString(lerr));
        return 1;
    }
    std::vector<std::uint16_t> host_out(kHeads * kD, 0);
    cudaMemcpy(host_out.data(), d_out, nbytes_in, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_pk);
    cudaFree(d_sc);

    // Compare per dimension. Tolerance = quantization step bound.
    // The stored code is round(code / scale) with |code|<=7, so the worst
    // reconstruction error is ~ scale (half a code). scale = amax/7; allow
    // a generous fraction of the group's amax plus bf16 rounding.
    double max_err = 0.0;
    int worst = -1;
    for (int head = 0; head < kHeads; ++head) {
        float amax_g[kGroups] = {0, 0, 0, 0};
        for (int g = 0; g < kGroups; ++g)
            for (int d = 0; d < 64; ++d) {
                const float v = bf16_to_f32(in_bits[head * kD + g * 64 + d]);
                amax_g[g] = fmaxf(amax_g[g], fabsf(v));
            }
        for (int d = 0; d < kD; ++d) {
            const float orig = bf16_to_f32(in_bits[head * kD + d]);
            const float got  = bf16_to_f32(host_out[head * kD + d]);
            const float scale = amax_g[d / 64] / 7.0f;
            // tolerance: half a code + fp16 scale rounding + fp32 roundtrip
            const float tol = 0.55f * (scale <= 0 ? 0.001f : scale) + 0.01f;
            const double e = fabs(static_cast<double>(got) - orig);
            if (e > max_err) { max_err = e; worst = head * kD + d; if (e > tol) std::printf("e>tol at head=%d d=%d orig=%.4f got=%.4f\n", head, d, orig, got); }
        }
    }
    std::printf("packed-V roundtrip max_abs_err=%.4f (worst_dim=%d) %s\n", max_err, worst,
                max_err < 3.0 ? "OK" : "FAIL");
    return max_err < 3.0 ? 0 : 1;
}