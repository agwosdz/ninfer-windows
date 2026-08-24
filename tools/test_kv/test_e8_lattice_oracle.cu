// Exact oracle for the production rk4v4-e8 codec (src/ops/kernel/e8_lattice.cuh):
//
//   1. The device E8 nearest-lattice projection (the warp path used by the
//      gqa_attention fill/decode kernels) must return the exact nearest point of
//      the Conway-Sloane E8 lattice, checked against an independent CPU reference
//      (FP64 algebraic decoder, cross-validated by an exhaustive exact
//      enumeration over both cosets).
//   2. The doubled code c = 2*p (the 4-bit code plus the coset bit carried by
//      parity) must be an exact integer in [-15, 15] and round-trip exactly:
//      c * 0.5f == p.
//   3. Codec quality: with the production pipeline (group scale amax/14,
//      projection domain k/(amax/7)), the exact E8 reconstruction must match
//      plain rk4v4 (4-bit, scale amax/7) within 1% on average and be far
//      better than the superseded half-coset approximation (rintf collapse of
//      the D8+1/2 coset). Both codecs quantize to the same step-(amax/7)
//      grid; E8 has the optimal 8-dim lattice shape but 1/16 the density, so
//      parity (not a win) is the expected relationship.
//
// The oracle is self-contained: it compiles the production device header
// directly and needs no engine, artifact, or host library. Quality statistics
// use FP32 group scales (production rounds the scale to FP16, which can only
// add error), so the measured improvement is an upper bound on the production
// margin.
//
// Sample conditioning. The exact-nearest comparison is only meaningful where
// the nearest E8 point is unambiguous beyond the FP32 resolution of the
// kernel's distance arithmetic, so samples are nudged off the Voronoi facets:
// no coordinate within 1e-3 of a multiple of 1/2 (within-coset ties) and the
// two coset distances separated by at least 1e-5 relative (inter-coset ties).
// The codec's exactness holds at every input; this is a test-domain
// well-conditioning, not a weakening of the property under test.

#include "ops/kernel/e8_lattice.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int kBlocks      = 200000; // exact-match + quality blocks
constexpr int kBruteBlocks = 512;    // brute-force cross-validation subset
constexpr int kDims        = 8;

// Mirrors the production gqa_attention store path: a 32-lane warp covers eight
// 8-dim blocks; each 8-lane subgroup holds one block per register (d0 = grp*64 +
// lane, d1 = d0 + 32 in the production layout). Projection then doubled code,
// exactly as the kernel computes them.
__global__ void oracle_kernel(const float* __restrict__ x, float* __restrict__ proj,
                              std::int8_t* __restrict__ code) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int block = warp * 8 + (lane >> 3) * 2; // kBlocks is a multiple of 8
    const int sub   = lane & 7;
    float v0 = x[static_cast<std::size_t>(block) * kDims + sub];
    float v1 = x[static_cast<std::size_t>(block + 1) * kDims + sub];
    ninfer::ops::e8_project_8d_warp(v0, v1, lane);
    proj[static_cast<std::size_t>(block) * kDims + sub]       = v0;
    proj[static_cast<std::size_t>(block + 1) * kDims + sub]   = v1;
    code[static_cast<std::size_t>(block) * kDims + sub]       = ninfer::ops::e8_doubled_code(v0);
    code[static_cast<std::size_t>(block + 1) * kDims + sub]   = ninfer::ops::e8_doubled_code(v1);
}

// ---------------------------------------------------------------------------
// Independent CPU references.
// ---------------------------------------------------------------------------

// Exact nearest E8 point: algebraic decoder in FP64 (D8 and D8+1/2 candidates,
// parity correction at the worst rounding coordinate, exact squared-distance
// selection). Returns false on a numeric tie (equidistant candidates), which
// the test treats as a failure because the codec's tie convention would be
// ambiguous to check against.
bool exact_e8_nearest_algebraic(const double x[kDims], double p[kDims]) {
    double f[kDims];
    long sum = 0;
    double max_err = -1.0;
    int worst      = 0;
    for (int i = 0; i < kDims; ++i) {
        f[i] = std::nearbyint(x[i]); // IEEE round-half-to-even, same convention as rintf
        const double err = std::abs(x[i] - f[i]);
        sum += static_cast<long>(f[i]);
        if (err > max_err) { max_err = err; worst = i; }
    }
    double d8[kDims];
    for (int i = 0; i < kDims; ++i) { d8[i] = f[i]; }
    if (sum % 2 != 0) { d8[worst] += (x[worst] >= f[worst]) ? 1.0 : -1.0; }

    double g[kDims];
    long sum2 = 0;
    double max_err2 = -1.0;
    int worst2      = 0;
    for (int i = 0; i < kDims; ++i) {
        const double xs = x[i] - 0.5;
        g[i]            = std::nearbyint(xs);
        const double err = std::abs(xs - g[i]);
        sum2 += static_cast<long>(g[i]);
        if (err > max_err2) { max_err2 = err; worst2 = i; }
    }
    double c1[kDims];
    for (int i = 0; i < kDims; ++i) { c1[i] = g[i] + 0.5; }
    if (sum2 % 2 != 0) { c1[worst2] += ((x[worst2] - 0.5) >= g[worst2]) ? 1.0 : -1.0; }

    double dd = 0.0, dc = 0.0;
    for (int i = 0; i < kDims; ++i) {
        const double a = x[i] - d8[i];
        const double b = x[i] - c1[i];
        dd += a * a;
        dc += b * b;
    }
    if (std::abs(dd - dc) < 1e-12 * (dd > dc ? dd : dc)) { return false; }
    for (int i = 0; i < kDims; ++i) { p[i] = (dd <= dc) ? d8[i] : c1[i]; }
    return true;
}

// Relative separation of the two corrected coset distances (the inter-coset
// Voronoi facet distance), for sample conditioning.
double coset_gap(const double x[kDims]) {
    double f[kDims], g[kDims];
    long sum = 0, sum2 = 0;
    double max_err = -1.0, max_err2 = -1.0;
    int worst = 0, worst2 = 0;
    for (int i = 0; i < kDims; ++i) {
        f[i] = std::nearbyint(x[i]);
        sum += static_cast<long>(f[i]);
        if (std::abs(x[i] - f[i]) > max_err) { max_err = std::abs(x[i] - f[i]); worst = i; }
    }
    double dd = 0.0;
    for (int i = 0; i < kDims; ++i) {
        double v = f[i];
        if (sum % 2 != 0 && i == worst) { v += (x[worst] >= f[worst]) ? 1.0 : -1.0; }
        const double a = x[i] - v;
        dd += a * a;
    }
    for (int i = 0; i < kDims; ++i) {
        const double xs = x[i] - 0.5;
        g[i] = std::nearbyint(xs);
        sum2 += static_cast<long>(g[i]);
        if (std::abs(xs - g[i]) > max_err2) { max_err2 = std::abs(xs - g[i]); worst2 = i; }
    }
    double dc = 0.0;
    for (int i = 0; i < kDims; ++i) {
        double v = g[i] + 0.5;
        if (sum2 % 2 != 0 && i == worst2) { v += ((x[worst2] - 0.5) >= g[worst2]) ? 1.0 : -1.0; }
        const double b = x[i] - v;
        dc += b * b;
    }
    const double m = dd > dc ? dd : dc;
    return m > 0.0 ? std::abs(dd - dc) / m : 0.0;
}

// Exact nearest E8 point: brute-force enumeration, structurally independent of
// the algebraic decoder. E8 = {z in Z^8 : sum(z) even} union {z + 1/2 in
// (Z+1/2)^8 : sum(z) even}; the nearest point lies within the covering radius
// sqrt(2) of x, so the box z_i in [floor(x_i)-1, ceil(x_i)+1] per dimension is
// exhaustive. Returns false only on a bit-exact equidistant pair at the minimum
// (a genuine degeneracy; cannot occur on the conditioned samples).
bool exact_e8_nearest_bruteforce(const double x[kDims], double p[kDims]) {
    double vals[kDims][4];
    int cnt[kDims];
    for (int i = 0; i < kDims; ++i) {
        const double fl = std::floor(x[i]);
        const double ce = std::ceil(x[i]);
        int n          = 0;
        vals[i][n++]   = fl - 1.0;
        vals[i][n++]   = fl;
        if (ce != fl) { vals[i][n++] = ce; }
        vals[i][n++]   = ce + 1.0;
        cnt[i]         = n;
    }
    double best[2]         = {DBL_MAX, DBL_MAX};
    double bestz[2][kDims] = {{0}};
    bool tie[2]            = {false, false};
    long total             = 1;
    for (int i = 0; i < kDims; ++i) { total *= cnt[i]; }
    for (long c = 0; c < total; ++c) {
        long rem   = c;
        long zsum  = 0;
        double dist0 = 0.0, dist1 = 0.0;
        double z[kDims];
        for (int i = 0; i < kDims; ++i) {
            const int d = static_cast<int>(rem % cnt[i]);
            rem /= cnt[i];
            z[i] = vals[i][d];
            zsum += static_cast<long>(z[i]);
            const double a = x[i] - z[i];
            const double b = x[i] - (z[i] + 0.5);
            dist0 += a * a;
            dist1 += b * b;
        }
        if (zsum % 2 != 0) { continue; } // not an E8 integer part
        for (int s = 0; s < 2; ++s) {
            const double dist = (s == 0) ? dist0 : dist1;
            if (dist < best[s]) {
                best[s] = dist;
                tie[s]  = false;
                for (int i = 0; i < kDims; ++i) { bestz[s][i] = z[i]; }
            } else if (dist == best[s]) {
                tie[s] = true; // bit-exact equidistant candidate: genuine tie
            }
        }
    }
    if (tie[0] || tie[1]) { return false; }
    const int coset = (best[0] <= best[1]) ? 0 : 1;
    for (int i = 0; i < kDims; ++i) { p[i] = bestz[coset][i] + (coset == 1 ? 0.5 : 0.0); }
    return true;
}

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

template <typename T>
bool cuda_check(const char* what, T result) {
    if (result == cudaSuccess) { return true; }
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(result));
    return false;
}

int run() {
    // --- Synthetic blocks in the production codec domain. b ~ N(0, 0.2) per
    // coordinate; the group scale is amax/14 and the projection domain is
    // k/(amax/7) = k * (0.5f * kinv), kinv = 1/ks, matching the kernel pipeline.
    // x (the domain values) are then conditioned off the Voronoi facets.
    std::mt19937_64 rng(0x5EED1234ULL);
    std::normal_distribution<float> norm(0.0f, 0.2f);
    std::vector<float> v(static_cast<std::size_t>(kBlocks) * kDims);
    for (int b = 0; b < kBlocks; ++b) {
        for (int i = 0; i < kDims; ++i) {
            v[static_cast<std::size_t>(b) * kDims + i] = norm(rng);
        }
    }
    std::vector<float> ks_vec(kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < kDims; ++i) {
            const float a = std::abs(v[static_cast<std::size_t>(b) * kDims + i]);
            amax          = a > amax ? a : amax;
        }
        ks_vec[b] = amax / 14.0f;
    }
    std::vector<float> x(v.size());
    for (int b = 0; b < kBlocks; ++b) {
        const float k_inv = ks_vec[b] > 0.0f ? 1.0f / ks_vec[b] : 0.0f;
        for (int i = 0; i < kDims; ++i) {
            x[static_cast<std::size_t>(b) * kDims + i] =
                v[static_cast<std::size_t>(b) * kDims + i] * (0.5f * k_inv);
        }
    }

    // --- Conditioning: off the Voronoi facets (see file header).
    for (int b = 0; b < kBlocks; ++b) {
        std::vector<float> xb(x.begin() + static_cast<std::size_t>(b) * kDims,
                              x.begin() + static_cast<std::size_t>(b) * kDims + kDims);
        for (int i = 0; i < kDims; ++i) {
            const double m = std::round(xb[i] / 0.5);
            if (std::abs(xb[i] - 0.5 * m) < 1e-3) { xb[i] += (xb[i] >= 0.5 * m) ? 1e-3f : -1e-3f; }
        }
        for (int iter = 0; iter < 64; ++iter) {
            double xd[kDims];
            for (int i = 0; i < kDims; ++i) { xd[i] = static_cast<double>(xb[i]); }
            if (coset_gap(xd) >= 1e-5) { break; }
            xb[0] += 1e-4f;
            const double m0 = std::round(xb[0] / 0.5);
            if (std::abs(xb[0] - 0.5 * m0) < 1e-3) { xb[0] += 1e-3f; }
        }
        double xd[kDims];
        for (int i = 0; i < kDims; ++i) { xd[i] = static_cast<double>(xb[i]); }
        if (coset_gap(xd) < 1e-5) {
            std::fprintf(stderr, "FAIL: could not condition block %d off the inter-coset facet\n", b);
            return 1;
        }
        for (int i = 0; i < kDims; ++i) {
            x[static_cast<std::size_t>(b) * kDims + i] = xb[i];
        }
    }

    // --- Device run (production sequence: warp projection then doubled code).
    float *d_x = nullptr, *d_pw = nullptr;
    std::int8_t* d_code = nullptr;
    if (!cuda_check("alloc", cudaMalloc(&d_x, x.size() * sizeof(float))) ||
        !cuda_check("alloc", cudaMalloc(&d_pw, x.size() * sizeof(float))) ||
        !cuda_check("alloc", cudaMalloc(&d_code, x.size() * sizeof(std::int8_t)))) {
        return 1;
    }
    if (!cuda_check("memcpy x", cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                                           cudaMemcpyHostToDevice))) {
        return 1;
    }
    const int blocks_per_cta = 64; // 8 warps x 8 blocks
    oracle_kernel<<<kBlocks / blocks_per_cta, 256>>>(d_x, d_pw, d_code);
    if (!cuda_check("kernel", cudaGetLastError()) ||
        !cuda_check("sync", cudaDeviceSynchronize())) {
        return 1;
    }
    std::vector<float> pw(x.size());
    std::vector<std::int8_t> code(x.size());
    if (!cuda_check("memcpy out", cudaMemcpy(pw.data(), d_pw, pw.size() * sizeof(float),
                                             cudaMemcpyDeviceToHost)) ||
        !cuda_check("memcpy out", cudaMemcpy(code.data(), d_code, code.size() * sizeof(std::int8_t),
                                             cudaMemcpyDeviceToHost))) {
        return 1;
    }

    // --- CPU references.
    std::vector<double> ref(x.size());
    for (int b = 0; b < kBlocks; ++b) {
        double xd[kDims], p[kDims];
        for (int i = 0; i < kDims; ++i) {
            xd[i] = static_cast<double>(x[static_cast<std::size_t>(b) * kDims + i]);
        }
        if (!exact_e8_nearest_algebraic(xd, p)) {
            std::fprintf(stderr, "FAIL: algebraic reference hit a numeric tie at block %d\n", b);
            return 1;
        }
        for (int i = 0; i < kDims; ++i) {
            ref[static_cast<std::size_t>(b) * kDims + i] = p[i];
        }
        if (b < kBruteBlocks) {
            double pb[kDims];
            if (!exact_e8_nearest_bruteforce(xd, pb)) {
                std::fprintf(stderr, "FAIL: brute-force reference hit a numeric tie at block %d\n", b);
                return 1;
            }
            for (int i = 0; i < kDims; ++i) {
                if (std::llround(p[i] * 2.0) != std::llround(pb[i] * 2.0)) {
                    std::fprintf(stderr,
                                 "FAIL: algebraic and brute-force references disagree at block %d dim %d\n",
                                 b, i);
                    return 1;
                }
            }
        }
    }

    // --- Exact-match checks: device == exact nearest E8 point, codes exact.
    long mismatch = 0, code_bad = 0, parity_bad = 0, roundtrip_bad = 0;
    for (int b = 0; b < kBlocks; ++b) {
        const std::size_t o = static_cast<std::size_t>(b) * kDims;
        long two_p0 = 0;
        for (int i = 0; i < kDims; ++i) {
            const double pdev = static_cast<double>(pw[o + i]);
            const long two_p  = std::llround(pdev * 2.0);
            if (std::abs(pdev * 2.0 - static_cast<double>(two_p)) > 1e-5) { ++mismatch; }
            if (two_p != std::llround(ref[o + i] * 2.0)) { ++mismatch; }
            if (i == 0) { two_p0 = two_p; }
            if ((two_p & 1) != (two_p0 & 1)) { ++parity_bad; }
        }
        for (int i = 0; i < kDims; ++i) {
            const int c = code[o + i];
            if (c != static_cast<int>(pw[o + i] * 2.0f) || c < -15 || c > 15) { ++code_bad; }
            if (static_cast<float>(c) * 0.5f != pw[o + i]) { ++roundtrip_bad; }
        }
    }
    if (mismatch || code_bad || parity_bad || roundtrip_bad) {
        std::fprintf(stderr,
                     "FAIL: exact codec checks (projection mismatch=%ld, code range/round=%ld, "
                     "coset parity=%ld, float round-trip=%ld)\n",
                     mismatch, code_bad, parity_bad, roundtrip_bad);
        return 1;
    }
    std::printf("  [PASS] device E8 warp projection == exact nearest lattice point (%d blocks)\n",
                kBlocks);
    std::printf("  [PASS] doubled codes exact in [-15,15], parity = coset, c*0.5f == p (%d blocks)\n",
                kBlocks);
    std::printf("  [PASS] brute-force enumeration cross-validates the algebraic reference (%d blocks)\n",
                kBruteBlocks);

    // --- Quality statistics (production pipeline, FP32 scales).
    double sum_cos_e8 = 0.0, sum_rel_e8 = 0.0, sum_cos_plain = 0.0, sum_rel_plain = 0.0;
    double sum_rel_old = 0.0;
    for (int b = 0; b < kBlocks; ++b) {
        const std::size_t o = static_cast<std::size_t>(b) * kDims;
        const float ks  = ks_vec[b];
        const float ks7 = ks * 2.0f; // amax/7, the plain rk4v4 group scale
        double nb = 0.0, ee8 = 0.0, epl = 0.0, eold = 0.0;
        double de8 = 0.0, dpl = 0.0, ne8 = 0.0, npl = 0.0;
        for (int i = 0; i < kDims; ++i) {
            const float k    = v[o + i];
            const double p   = ref[o + i];
            const float c    = static_cast<float>(code[o + i]);
            const float ve8  = c * ks; // exact E8: 2p * (amax/14)
            const long c4l   = std::llround(static_cast<double>(k / ks7));
            const int c4     = (c4l < -7) ? -7 : (c4l > 7) ? 7 : static_cast<int>(c4l);
            const float vpl  = static_cast<float>(c4) * ks7; // plain rk4v4
            const long col_l = std::llround(p);
            const float col  =
                static_cast<float>((col_l < -8) ? -8 : (col_l > 7) ? 7 : col_l) * ks7; // old half-coset
            nb += static_cast<double>(k) * k;
            const double e1 = ve8 - k, e2 = vpl - k, e3 = col - k;
            ee8 += e1 * e1;
            epl += e2 * e2;
            eold += e3 * e3;
            de8 += static_cast<double>(ve8) * k;
            dpl += static_cast<double>(vpl) * k;
            ne8 += static_cast<double>(ve8) * ve8;
            npl += static_cast<double>(vpl) * vpl;
        }
        const auto rel = [](double vv) { return std::sqrt(vv); };
        sum_rel_e8 += rel(ee8 / nb);
        sum_rel_plain += rel(epl / nb);
        sum_rel_old += rel(eold / nb);
        sum_cos_e8 += de8 / (std::sqrt(ne8 * nb) + 1e-30);
        sum_cos_plain += dpl / (std::sqrt(npl * nb) + 1e-30);
    }
    const double m = static_cast<double>(kBlocks);
    const double cos_e8 = sum_cos_e8 / m, rel_e8 = sum_rel_e8 / m;
    const double cos_pl = sum_cos_plain / m, rel_pl = sum_rel_plain / m;
    std::printf("  rk4v4-e8 (exact E8, this codec) : mean cos %.5f, mean rel RMS %.5f\n",
                cos_e8, rel_e8);
    std::printf("  rk4v4    (plain int4 K)         : mean cos %.5f, mean rel RMS %.5f\n",
                cos_pl, rel_pl);
    std::printf("  old half-coset (rintf collapse) : mean rel RMS %.5f\n", sum_rel_old / m);
    const double rel_old = sum_rel_old / m;
    if (!(rel_e8 < rel_old)) {
        std::fprintf(stderr,
                     "FAIL: exact E8 codec is not more accurate than the superseded half-coset "
                     "(%.5f >= %.5f)\n",
                     rel_e8, rel_old);
        return 1;
    }
    if (!(rel_e8 <= 1.01 * rel_pl)) {
        std::fprintf(stderr, "FAIL: exact E8 codec is not within 1%% of plain rk4v4 quality (%.5f > %.5f)\n",
                     rel_e8, 1.01 * rel_pl);
        return 1;
    }
    std::printf("  [PASS] exact E8 matches plain rk4v4 within 1%% (rel RMS %.5f <= %.5f)\n", rel_e8,
                1.01 * rel_pl);
    std::printf("  [PASS] exact E8 beats the superseded half-coset (rel RMS %.5f < %.5f, ~%.0f%% better)\n",
                rel_e8, rel_old, (1.0 - rel_e8 / rel_old) * 100.0);
    return 0;
}

} // namespace

int main() { return ::run(); }
