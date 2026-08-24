#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

int verify_guards(const std::string& label, const GuardedDeviceBuffer& buffer) {
    return buffer.verify_guards(label.c_str());
}

// The drafter's two-tap grouped depthwise conv accumulates in FP32 and stores
// BF16; the oracle evaluates the same formula in double over the bf16-rounded
// logical inputs. Divergence is the BF16 output store plus FP32 intermediate
// products, so the causal-conv-style reduction bound applies.
constexpr ReductionCriterion kDflash2ConvCriterion{
    /*relative_l2*/ 1.85e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 3.7e-3,
};

constexpr std::uint8_t kOutputPoison = 0xff;

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { result[i] = f32_to_bf16(values[i]); }
    return result;
}

std::vector<float> make_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return values;
}

// Oracle for the two-tap grouped depthwise causal conv in double over the
// bf16-rounded logical inputs x, dynamic, base. Layouts are dim0-fastest.
//
//   out[c, t] = (dynamic[g + G*0 + G*2*side, t] + base[side,0,c]) * x[c,t]
//             + (dynamic[g + G*1 + G*2*side, t] + base[side,1,c]) * (j>=1 ? x[c,t-1] : 0)
//   g = c / 16, G = hidden / 16, j = t % width (block-local causal shift).
std::vector<double> conv_oracle(const std::vector<float>& x, const std::vector<float>& dynamic,
                                const std::vector<float>& base, std::int32_t hidden,
                                std::int32_t tokens, std::int32_t width, std::int32_t side) {
    const std::int32_t groups    = hidden / kDFlash2ConvGroupSize;
    const std::int32_t dyn_rows  = kDFlash2ConvKernel * 2 * groups;
    const std::int32_t base_off0 = side * (kDFlash2ConvKernel * hidden);
    const std::int32_t base_off1 = hidden + base_off0;
    const std::int32_t dyn_row0  = side * (kDFlash2ConvKernel * groups);
    const std::int32_t dyn_row1  = groups + dyn_row0;

    std::vector<double> out(static_cast<std::size_t>(hidden) * tokens, 0.0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t j = t % width;
        for (std::int32_t c = 0; c < hidden; ++c) {
            const std::size_t xj       = static_cast<std::size_t>(c) + static_cast<std::size_t>(hidden) * t;
            const std::int32_t g       = c / kDFlash2ConvGroupSize;
            const double x0            = static_cast<double>(x[xj]);
            const double x1            = j >= 1 ? static_cast<double>(x[xj - static_cast<std::size_t>(hidden)]) : 0.0;
            const double w0            = static_cast<double>(dynamic[static_cast<std::size_t>(dyn_row0) + static_cast<std::size_t>(dyn_rows) * t + g]) +
                                         static_cast<double>(base[static_cast<std::size_t>(base_off0) + c]);
            const double w1            = static_cast<double>(dynamic[static_cast<std::size_t>(dyn_row1) + static_cast<std::size_t>(dyn_rows) * t + g]) +
                                         static_cast<double>(base[static_cast<std::size_t>(base_off1) + c]);
            out[xj]                   = w0 * x0 + w1 * x1;
        }
    }
    return out;
}

int conv_case(std::int32_t hidden, std::int32_t width, std::int32_t batch, std::int32_t side,
              std::uint32_t seed) {
    const std::int32_t tokens    = width * batch;
    const std::int32_t groups    = hidden / kDFlash2ConvGroupSize;
    const std::int32_t dyn_rows  = kDFlash2ConvKernel * 2 * groups;
    const std::int32_t base_cols = kDFlash2ConvKernel * 2 * hidden;

    std::vector<float> x       = make_values(static_cast<std::size_t>(hidden) * tokens, seed, -3.0F, 3.0F);
    std::vector<float> dynamic = make_values(static_cast<std::size_t>(dyn_rows) * tokens, seed + 1U, -1.0F, 1.0F);
    std::vector<float> base    = make_values(static_cast<std::size_t>(base_cols), seed + 2U, -1.0F, 1.0F);

    // Deterministic cancellation/sign channels: opposite tap weights over the
    // group's channel against alternating per-token values. Injects a clear
    // grouping/u/order signal independent of the oracle's uniform draw.
    for (const std::int32_t c : {0, hidden - 1, hidden / 2, hidden / 7}) {
        base[static_cast<std::size_t>(side * (kDFlash2ConvKernel * hidden)) + c]        = 1.0F;
        base[static_cast<std::size_t>(side * (kDFlash2ConvKernel * hidden)) + hidden + c] = -1.0F;
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::int32_t j = t % width;
            x[static_cast<std::size_t>(c) + static_cast<std::size_t>(hidden) * t] =
                (j & 1) == 0 ? 0.75F : -0.75F;
        }
    }

    const std::vector<double> oracle = conv_oracle(x, dynamic, base, hidden, tokens, width, side);
    const std::vector<std::uint16_t> x_bits = bf16_bits(x);
    const std::vector<std::uint16_t> dynamic_bits = bf16_bits(dynamic);
    const std::vector<std::uint16_t> base_bits = bf16_bits(base);

    GuardedDeviceBuffer dx(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dd(dynamic_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer db(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dout(x_bits.size() * sizeof(std::uint16_t));
    dx.copy_from_host(x_bits.data(), dx.bytes());
    dd.copy_from_host(dynamic_bits.data(), dd.bytes());
    db.copy_from_host(base_bits.data(), db.bytes());
    dout.fill(kOutputPoison);

    Tensor tx(dx.data(), DType::BF16, {hidden, tokens});
    Tensor td(dd.data(), DType::BF16, {dyn_rows, tokens});
    Tensor tb(db.data(), DType::BF16, {kDFlash2ConvKernel, 2, hidden});
    Tensor tout(dout.data(), DType::BF16, {hidden, tokens});

    ops::dflash2_dynamic_conv(tx, td, tb, side, width, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "dflash2_dynamic_conv hidden=" + std::to_string(hidden) +
                            " width=" + std::to_string(width) + " batch=" + std::to_string(batch) +
                            " side=" + std::to_string(side);
    int failures = 0;
    failures += verify_reduction(tag + " output", from_device_bf16(dout.data(), x_bits.size()), oracle,
                                 kDflash2ConvCriterion);
    failures += verify_guards(tag + " x", dx);
    failures += verify_guards(tag + " dynamic", dd);
    failures += verify_guards(tag + " base", db);
    failures += verify_guards(tag + " output", dout);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // The 27B dflash2 hidden extent (5120) with block 8; cover both sides,
    // single-block, multi-block, and batch>1.
    constexpr std::int32_t kQwen8Hidden = 5120;
    for (const std::int32_t side : {0, 1}) {
        failures += conv_case(kQwen8Hidden, 1, 1, side, 100U + static_cast<std::uint32_t>(side));
        failures += conv_case(kQwen8Hidden, 8, 1, side, 108U + static_cast<std::uint32_t>(side));
        failures += conv_case(kQwen8Hidden, 8, 3, side, 183U + static_cast<std::uint32_t>(side));
        failures += conv_case(kQwen8Hidden, 16, 1, side, 116U + static_cast<std::uint32_t>(side));
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash2_dynamic_conv\n";
    return failures == 0 ? 0 : 1;
}