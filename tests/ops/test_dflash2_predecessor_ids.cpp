#include "ninfer/ops/dflash2_predecessor_ids.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

int verify_guards(const std::string& label, const GuardedDeviceBuffer& buffer) {
    return buffer.verify_guards(label.c_str());
}

// Exact integer transform. The oracle shifts each candidate list one block
// column back and broadcasts the anchor over the first scored positions:
//
//   out[k, t], t = b * block_tokens + pos
//     pos == 0 : anchor_ids[b]
//     pos == 1 : anchor_ids[b]
//     pos >= 2 : candidate_ids[k, t - 1]
//
// Layouts are dim0-fastest I32: flat index = k + top_k * t.
std::vector<std::int32_t> predecessor_oracle(const std::vector<std::int32_t>& candidate_ids,
                                             const std::vector<std::int32_t>& anchor_ids,
                                             std::int32_t top_k, std::int32_t tokens,
                                             std::int32_t block_tokens) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(top_k) * tokens, 0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t pos  = t % block_tokens;
        const std::int32_t b    = t / block_tokens;
        const std::int32_t src  = t - 1;
        for (std::int32_t k = 0; k < top_k; ++k) {
            std::int32_t value = anchor_ids[b];
            if (pos >= 2) {
                value = candidate_ids[static_cast<std::size_t>(k) + static_cast<std::size_t>(top_k) * src];
            }
            out[static_cast<std::size_t>(k) + static_cast<std::size_t>(top_k) * t] = value;
        }
    }
    return out;
}

int predecessor_case(std::int32_t top_k, std::int32_t block_tokens, std::int32_t batch,
                     std::uint32_t seed) {
    const std::int32_t tokens = block_tokens * batch;
    std::vector<std::int32_t> candidate_ids(static_cast<std::size_t>(top_k) * tokens);
    for (std::size_t i = 0; i < candidate_ids.size(); ++i) {
        candidate_ids[i] = static_cast<std::int32_t>(1000 + i + (seed % 4096u));
    }
    // Force the block-boundary interplay: put an identifiable marker at the
    // first candidate of every column so a wrong source column is obvious.
    for (std::int32_t t = 0; t < tokens; ++t) {
        candidate_ids[static_cast<std::size_t>(t) * top_k] = 500 + t;
    }
    std::vector<std::int32_t> anchor_ids(batch);
    for (std::int32_t b = 0; b < batch; ++b) { anchor_ids[b] = 9000 + b; }

    const std::vector<std::int32_t> oracle =
        predecessor_oracle(candidate_ids, anchor_ids, top_k, tokens, block_tokens);

    GuardedDeviceBuffer dc(candidate_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer da(anchor_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(candidate_ids.size() * sizeof(std::int32_t));
    dc.copy_from_host(candidate_ids.data(), dc.bytes());
    da.copy_from_host(anchor_ids.data(), da.bytes());
    dout.fill(0x5a);

    Tensor tc(dc.data(), DType::I32, {top_k, tokens});
    Tensor ta(da.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::I32, {top_k, tokens});

    ops::dflash2_predecessor_ids(tc, ta, block_tokens, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "dflash2_predecessor_ids top_k=" + std::to_string(top_k) +
                            " block=" + std::to_string(block_tokens) + " batch=" + std::to_string(batch);
    int failures = 0;
    failures += verify_exact((tag + " output").c_str(), from_device<std::int32_t>(dout.data(), oracle.size()), oracle);
    failures += verify_exact((tag + " candidate_ids preserved").c_str(), from_device<std::int32_t>(dc.data(), candidate_ids.size()), candidate_ids);
    failures += verify_exact((tag + " anchor_ids preserved").c_str(), from_device<std::int32_t>(da.data(), anchor_ids.size()), anchor_ids);
    failures += verify_guards(tag + " candidate_ids", dc);
    failures += verify_guards(tag + " anchor_ids", da);
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
    constexpr std::int32_t kQwen8TopK = 16;
    failures += predecessor_case(kQwen8TopK, 8, 1, 11U);
    failures += predecessor_case(kQwen8TopK, 8, 3, 83U);
    failures += predecessor_case(kQwen8TopK, 16, 1, 161U);
    failures += predecessor_case(kQwen8TopK, 8, 5, 85U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash2_predecessor_ids\n";
    return failures == 0 ? 0 : 1;
}