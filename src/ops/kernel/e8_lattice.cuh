#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace ninfer::ops {

// 8x8 Sylvester-Hadamard orthogonal rotation in CUDA registers
// Multiplies vector by normalized Hadamard matrix H_8 / sqrt(8)
__device__ __forceinline__ void hadamard_rot_8d(const float in[8], float out[8]) {
    constexpr float kInvSqrt8 = 0.35355339059327373f; // 1/sqrt(8)
    
    // Fast in-place butterfly stages
    float a0 = in[0] + in[1]; float a1 = in[0] - in[1];
    float a2 = in[2] + in[3]; float a3 = in[2] - in[3];
    float a4 = in[4] + in[5]; float a5 = in[4] - in[5];
    float a6 = in[6] + in[7]; float a7 = in[6] - in[7];

    float b0 = a0 + a2; float b1 = a1 + a3;
    float b2 = a0 - a2; float b3 = a1 - a3;
    float b4 = a4 + a6; float b5 = a5 + a7;
    float b6 = a4 - a6; float b7 = a5 - a7;

    out[0] = (b0 + b4) * kInvSqrt8;
    out[1] = (b1 + b5) * kInvSqrt8;
    out[2] = (b2 + b6) * kInvSqrt8;
    out[3] = (b3 + b7) * kInvSqrt8;
    out[4] = (b0 - b4) * kInvSqrt8;
    out[5] = (b1 - b5) * kInvSqrt8;
    out[6] = (b2 - b6) * kInvSqrt8;
    out[7] = (b3 - b7) * kInvSqrt8;
}

// rk4v4-e8 codec contract: the group scale is amax/14, the projection domain is
// k/(amax/7) = k * (0.5 / (amax/14)) so |x| <= 7 per coordinate. The nearest E8
// point p is therefore an integer or half-integer with |p| <= 7.5, and the
// doubled code c = 2*p is an exact integer in [-15, 15] (4-bit code + coset bit;
// the coset is carried by code parity). Storing c per dimension as int8 with the
// group scale amax/14 makes the generic int8 dequant exact:
//   c * (amax/14) == 2*p * (amax/14) == p * (amax/7).
__device__ __forceinline__ std::int8_t e8_doubled_code(float p) {
    int c = static_cast<int>(rintf(p * 2.0f));
    c     = max(-15, min(15, c)); // defensive; in range for |p| <= 7.5
    return static_cast<std::int8_t>(c);
}

// Warp-Cooperative 8D E8 Projection across 8 lanes in a 32-thread warp
__device__ __forceinline__ float e8_project_8d_warp_single(float x, int lane, unsigned sub_mask) {
    const int sub_lane = lane & 7;

    // 1. D8 Candidate
    float f = rintf(x);
    int sum_f = static_cast<int>(f);
    sum_f += __shfl_xor_sync(sub_mask, sum_f, 1);
    sum_f += __shfl_xor_sync(sub_mask, sum_f, 2);
    sum_f += __shfl_xor_sync(sub_mask, sum_f, 4);

    float max_err = fabsf(x - f);
    int worst_lane = sub_lane;
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        float other_err = __shfl_xor_sync(sub_mask, max_err, offset);
        int other_lane = __shfl_xor_sync(sub_mask, worst_lane, offset);
        if (other_err > max_err || (other_err == max_err && other_lane < worst_lane)) {
            max_err = other_err;
            worst_lane = other_lane;
        }
    }

    float d8 = f;
    if ((sum_f & 1) != 0 && sub_lane == worst_lane) {
        d8 += (x >= f) ? 1.0f : -1.0f;
    }

    // 2. Coset 1 Candidate: D8 + 0.5
    float xs = x - 0.5f;
    float f_s = rintf(xs);
    int sum_fs = static_cast<int>(f_s);
    sum_fs += __shfl_xor_sync(sub_mask, sum_fs, 1);
    sum_fs += __shfl_xor_sync(sub_mask, sum_fs, 2);
    sum_fs += __shfl_xor_sync(sub_mask, sum_fs, 4);

    float max_err_s = fabsf(xs - f_s);
    int worst_lane_s = sub_lane;
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        float other_err = __shfl_xor_sync(sub_mask, max_err_s, offset);
        int other_lane = __shfl_xor_sync(sub_mask, worst_lane_s, offset);
        if (other_err > max_err_s || (other_err == max_err_s && other_lane < worst_lane_s)) {
            max_err_s = other_err;
            worst_lane_s = other_lane;
        }
    }

    float coset1 = f_s + 0.5f;
    if ((sum_fs & 1) != 0 && sub_lane == worst_lane_s) {
        coset1 += (xs >= f_s) ? 1.0f : -1.0f;
    }

    // 3. Compare squared distances
    float diff0 = x - d8;
    float diff1 = x - coset1;
    float dist0 = diff0 * diff0;
    float dist1 = diff1 * diff1;
    dist0 += __shfl_xor_sync(sub_mask, dist0, 1);
    dist0 += __shfl_xor_sync(sub_mask, dist0, 2);
    dist0 += __shfl_xor_sync(sub_mask, dist0, 4);

    dist1 += __shfl_xor_sync(sub_mask, dist1, 1);
    dist1 += __shfl_xor_sync(sub_mask, dist1, 2);
    dist1 += __shfl_xor_sync(sub_mask, dist1, 4);

    return (dist0 <= dist1) ? d8 : coset1;
}

// Warp-Cooperative 8D E8 Projection for two 32-dim halves (d0, d1) in parallel
__device__ __forceinline__ void e8_project_8d_warp(float& x0, float& x1, int lane) {
    const unsigned sub_mask = 0xFFu << (lane & 24);
    x0 = e8_project_8d_warp_single(x0, lane, sub_mask);
    x1 = e8_project_8d_warp_single(x1, lane, sub_mask);
}

} // namespace ninfer::ops
