#include "ninfer/ops/dflash2_selector_lattice.h"
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

std::vector<double> f32_read(const void* device, std::size_t n) {
    const std::vector<float> f = from_device<float>(device, n);
    return std::vector<double>(f.begin(), f.end());
}

// Selector kernels accumulate the rank dot product in FP32 (the cores read
// BF16 operands), while the oracle uses double. Residual fp32-vs-double
// accumulation drift is ~1e-6 relative; the bound absorbs that and still
// catches a wrong index/order or a broken logit/weight source.
constexpr ReductionCriterion kLatticeReduction{
    /*relative_l2*/ 1.0e-4,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 1.0e-3,
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

float pred_bf16_value(float product) { return bf16_to_f32(f32_to_bf16(product)); }

// --- dflash2_select_candidates oracle ---------------------------------------
// Top-16 of each logit column by value desc, lower token id breaking ties.
void oracle_top_k(const std::vector<float>& column, std::int32_t k, std::vector<float>& best_values,
                  std::vector<std::int32_t>& best_ids) {
    best_values.assign(k, -std::numeric_limits<float>::infinity());
    best_ids.resize(k);
    std::fill(best_ids.begin(), best_ids.end(), 0);
    for (std::size_t id = 0; id < column.size(); ++id) {
        if (column[id] <= best_values[k - 1]) { continue; }
        std::int32_t slot = k - 1;
        while (slot > 0 &&
               (column[id] > best_values[slot - 1] ||
                (column[id] == best_values[slot - 1] && static_cast<std::int32_t>(id) < best_ids[slot - 1]))) {
            best_values[slot] = best_values[slot - 1];
            best_ids[slot]    = best_ids[slot - 1];
            --slot;
        }
        best_values[slot] = column[id];
        best_ids[slot]    = static_cast<std::int32_t>(id);
    }
}

int candidates_case(std::int32_t vocab, std::int32_t tokens, std::uint32_t seed) {
    const std::vector<float> logits =
        make_values(static_cast<std::size_t>(vocab) * tokens, seed, -12.0F, 8.0F);
    const std::vector<std::uint16_t> logits_bits = bf16_bits(logits);

    std::vector<std::int32_t> oracle_ids(static_cast<std::size_t>(kDFlash2SelectorTopK) * tokens);
    std::vector<float> oracle_values(static_cast<std::size_t>(kDFlash2SelectorTopK) * tokens, 0.0F);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::vector<float> col(logits.begin() + static_cast<std::size_t>(t) * vocab,
                                     logits.begin() + static_cast<std::size_t>(t + 1) * vocab);
        std::vector<float> best_values;
        std::vector<std::int32_t> best_ids;
        oracle_top_k(col, kDFlash2SelectorTopK, best_values, best_ids);
        for (std::int32_t s = 0; s < kDFlash2SelectorTopK; ++s) {
            oracle_ids[static_cast<std::size_t>(t) * kDFlash2SelectorTopK + s]   = best_ids[s];
            oracle_values[static_cast<std::size_t>(t) * kDFlash2SelectorTopK + s] = best_values[s];
        }
    }

    GuardedDeviceBuffer dlogits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dids(oracle_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvals(oracle_values.size() * sizeof(float));
    dlogits.copy_from_host(logits_bits.data(), dlogits.bytes());
    dids.fill(0x5a);
    dvals.fill(0x5a);

    Tensor tlogits(dlogits.data(), DType::BF16, {vocab, tokens});
    Tensor tids(dids.data(), DType::I32, {kDFlash2SelectorTopK, tokens});
    Tensor tvals(dvals.data(), DType::FP32, {kDFlash2SelectorTopK, tokens});

    ops::dflash2_select_candidates(tlogits, tids, tvals, nullptr);
    cuda_synchronize();

    const std::string tag = "dflash2_select_candidates vocab=" + std::to_string(vocab) +
                            " tokens=" + std::to_string(tokens);
    int failures = 0;
    const std::vector<std::int32_t> got_ids =
        from_device<std::int32_t>(dids.data(), oracle_ids.size());
    const std::vector<float> got_vals = from_device<float>(dvals.data(), oracle_values.size());
    {
        int printed = 0;
        for (std::size_t i = 0; i < got_ids.size() && printed < 8; ++i) {
            if (got_ids[i] != static_cast<std::int32_t>(oracle_ids[i])) {
                std::cerr << tag << " id mismatch at flat=" << i << " t=" << (i / kDFlash2SelectorTopK)
                          << " s=" << (i % kDFlash2SelectorTopK) << " got_id=" << got_ids[i]
                          << " exp_id=" << oracle_ids[i] << " got_val=" << got_vals[i]
                          << " exp_val=" << oracle_values[i] << '\n';
                const std::size_t col = i / kDFlash2SelectorTopK;
                for (std::int32_t s = 0; s < kDFlash2SelectorTopK; ++s) {
                    const std::size_t at = col * kDFlash2SelectorTopK + static_cast<std::size_t>(s);
                    std::cerr << "  s=" << s << " got(" << got_ids[at] << ", " << got_vals[at]
                              << ") exp(" << oracle_ids[at] << ", " << oracle_values[at] << ")\n";
                }
                std::cerr << "  boundary logits == " << oracle_values[i] << " at ids: ";
                for (std::int32_t id = 0; id < vocab; ++id) {
                    if (logits[static_cast<std::size_t>(col) * vocab + id] == oracle_values[i]) {
                        std::cerr << id << " ";
                    }
                }
                std::cerr << '\n';
                ++printed;
            }
        }
    }
    failures += verify_exact((tag + " ids").c_str(), got_ids, oracle_ids);
    failures += verify_reduction(tag + " values", f32_read(dvals.data(), oracle_values.size()),
                                 std::vector<double>(oracle_values.begin(), oracle_values.end()),
                                 kLatticeReduction);
    failures += verify_guards(tag + " logits", dlogits);
    failures += verify_guards(tag + " ids", dids);
    failures += verify_guards(tag + " values", dvals);
    return failures;
}

// --- dflash2_selector_lattice --------------------------------------------------
// Oracle for the packed row per column t (anchor columns stay zero):
//   out[row + c]
//     c < top_k                   : candidate ids (from `candidates`)
//     top_k <= c < top_k+top_k^2  : score[p][s], p=(c-k)/k, s=(c-k)%k
//         = sum_r successor[s,r,t] * bf16(predecessor[p,r,t]*hidden_pos[r,t])
//           + unary[s][t]
std::vector<float> lattice_oracle(const std::vector<float>& hidden_pos,
                                  const std::vector<float>& successor,
                                  const std::vector<float>& predecessor,
                                  const std::vector<std::int32_t>& candidates,
                                  const std::vector<float>& unary, std::int32_t rank,
                                  std::int32_t tokens, std::int32_t packed_width,
                                  std::int32_t block_tokens) {
    const std::int32_t k  = kDFlash2SelectorTopK;
    const std::int32_t k2 = k + k * k;
    std::vector<float> out(static_cast<std::size_t>(packed_width) * tokens, 0.0F);

    for (std::int32_t t = 0; t < tokens; ++t) {
        if (t % block_tokens == 0) { continue; }
        const std::size_t row = static_cast<std::size_t>(t) * packed_width;
        for (std::int32_t s = 0; s < k; ++s) {
            out[row + s] = static_cast<float>(candidates[static_cast<std::size_t>(t) * k + s]);
        }
        for (std::int32_t p = 0; p < k; ++p) {
            for (std::int32_t s = 0; s < k; ++s) {
                double score = 0.0;
                for (std::int32_t r = 0; r < rank; ++r) {
                    const float pred =
                        predecessor[static_cast<std::size_t>(t * k + p) * rank + r];
                    const float hid = hidden_pos[static_cast<std::size_t>(t) * rank + r];
                    const float succ =
                        successor[static_cast<std::size_t>(t * k + s) * rank + r];
                    score += static_cast<double>(succ) * pred_bf16_value(pred * hid);
                }
                score += unary[static_cast<std::size_t>(t) * k + s];
                const std::size_t c = static_cast<std::size_t>(k) + static_cast<std::size_t>(p * k + s);
                out[row + c] = static_cast<float>(score);
            }
        }
    }
    return out;
}

int tie_boundary_case(std::uint32_t seed) {
    // A tie-heavy column: one dominant value shared by many ids, straddling the
    // 16th candidate boundary. The global contract (value desc, lower id ties)
    // must put the LOWEST ids at the boundary; any per-thread pruning without
    // the id tie-break reorders them. 4992 vocab, every 100th logit is exactly
    // 4.0 (50 ids tied), the rest distinct lower values; the top quadrant is a
    // block of exact 8.0s so the boundary mixes value and id ties.
    constexpr std::int32_t kVocab = 4992;
    constexpr std::int32_t kTokens = 4;
    std::vector<float> logits(static_cast<std::size_t>(kVocab) * kTokens, -3.0F);
    fill_uniform(logits, seed, -6.0F, 3.0F);
    round_to_bf16(logits);
    for (std::int32_t t = 0; t < kTokens; ++t) {
        for (std::int32_t id = 0; id < 20; ++id) {
            logits[static_cast<std::size_t>(t) * kVocab + id] = 8.0F; // 20 exact ties at the top
        }
        for (std::int32_t id = 40; id < 40 + 60; ++id) {
            logits[static_cast<std::size_t>(t) * kVocab + id] = 4.0F; // 60 ties below the top
        }
    }

    const std::vector<std::uint16_t> logits_bits = bf16_bits(logits);
    std::vector<std::int32_t> oracle_ids(static_cast<std::size_t>(kDFlash2SelectorTopK) * kTokens);
    std::vector<float> oracle_values(static_cast<std::size_t>(kDFlash2SelectorTopK) * kTokens, 0.0F);
    for (std::int32_t t = 0; t < kTokens; ++t) {
        const std::vector<float> col(logits.begin() + static_cast<std::size_t>(t) * kVocab,
                                     logits.begin() + static_cast<std::size_t>(t + 1) * kVocab);
        std::vector<float> best_values;
        std::vector<std::int32_t> best_ids;
        oracle_top_k(col, kDFlash2SelectorTopK, best_values, best_ids);
        for (std::int32_t s = 0; s < kDFlash2SelectorTopK; ++s) {
            oracle_ids[static_cast<std::size_t>(t) * kDFlash2SelectorTopK + s] = best_ids[s];
            oracle_values[static_cast<std::size_t>(t) * kDFlash2SelectorTopK + s] = best_values[s];
        }
    }

    GuardedDeviceBuffer dlogits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dids(oracle_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvals(oracle_values.size() * sizeof(float));
    dlogits.copy_from_host(logits_bits.data(), dlogits.bytes());
    dids.fill(0x5a);
    dvals.fill(0x5a);

    Tensor tlogits(dlogits.data(), DType::BF16, {kVocab, kTokens});
    Tensor tids(dids.data(), DType::I32, {kDFlash2SelectorTopK, kTokens});
    Tensor tvals(dvals.data(), DType::FP32, {kDFlash2SelectorTopK, kTokens});
    ops::dflash2_select_candidates(tlogits, tids, tvals, nullptr);
    cuda_synchronize();

    const std::string tag = "dflash2_select_candidates boundary-ties";
    int failures = 0;
    failures += verify_exact((tag + " ids").c_str(), from_device<std::int32_t>(dids.data(), oracle_ids.size()),
                             oracle_ids);
    failures += verify_reduction(tag + " values", f32_read(dvals.data(), oracle_values.size()),
                                 std::vector<double>(oracle_values.begin(), oracle_values.end()),
                                 kLatticeReduction);
    failures += verify_guards(tag + " logits", dlogits);
    failures += verify_guards(tag + " ids", dids);
    failures += verify_guards(tag + " values", dvals);
    return failures;
}

int lattice_case(std::int32_t rank, std::int32_t packed_width, std::int32_t block_tokens,
                 std::int32_t batch, std::uint32_t seed) {
    const std::int32_t k       = kDFlash2SelectorTopK;
    const std::int32_t tokens  = block_tokens * batch;

    const std::vector<float> hidden_pos  = make_values(static_cast<std::size_t>(rank) * tokens, seed, -1.0F, 1.0F);
    const std::vector<float> successor   = make_values(static_cast<std::size_t>(rank) * k * tokens, seed + 1U, -1.0F, 1.0F);
    const std::vector<float> predecessor = make_values(static_cast<std::size_t>(rank) * k * tokens, seed + 2U, -1.0F, 1.0F);
    const std::vector<float> unary       = make_values(static_cast<std::size_t>(k) * tokens, seed + 3U, -3.0F, 3.0F);

    std::vector<std::int32_t> candidates(static_cast<std::size_t>(k) * tokens);
    for (std::size_t i = 0; i < candidates.size(); ++i) { candidates[i] = static_cast<std::int32_t>(i + 17); }

    const std::vector<float> oracle =
        lattice_oracle(hidden_pos, successor, predecessor, candidates, unary, rank, tokens,
                       packed_width, block_tokens);

    GuardedDeviceBuffer dh(hidden_pos.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer ds(successor.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(predecessor.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dc(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer du(unary.size() * sizeof(float));
    GuardedDeviceBuffer dout(oracle.size() * sizeof(float));
    const std::vector<std::uint16_t> hb = bf16_bits(hidden_pos);
    const std::vector<std::uint16_t> sb = bf16_bits(successor);
    const std::vector<std::uint16_t> pb = bf16_bits(predecessor);
    dh.copy_from_host(hb.data(), dh.bytes());
    ds.copy_from_host(sb.data(), ds.bytes());
    dp.copy_from_host(pb.data(), dp.bytes());
    dc.copy_from_host(candidates.data(), dc.bytes());
    du.copy_from_host(unary.data(), du.bytes());
    dout.fill(0x5a);

    Tensor th(dh.data(), DType::BF16, {rank, tokens});
    Tensor ts(ds.data(), DType::BF16, {rank, k, tokens});
    Tensor tp(dp.data(), DType::BF16, {rank, k, tokens});
    Tensor tc(dc.data(), DType::I32, {k, tokens});
    Tensor tu(du.data(), DType::FP32, {k, tokens});
    Tensor tout(dout.data(), DType::FP32, {packed_width, tokens});

    ops::dflash2_selector_lattice(th, ts, tp, tc, tu, packed_width, block_tokens, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "dflash2_selector_lattice rank=" + std::to_string(rank) +
                            " packed=" + std::to_string(packed_width) +
                            " block=" + std::to_string(block_tokens) +
                            " batch=" + std::to_string(batch);
    int failures = 0;
    failures += verify_reduction(tag + " output", f32_read(dout.data(), oracle.size()),
                                 std::vector<double>(oracle.begin(), oracle.end()), kLatticeReduction);
    failures += verify_guards(tag + " hidden_pos", dh);
    failures += verify_guards(tag + " successor", ds);
    failures += verify_guards(tag + " predecessor", dp);
    failures += verify_guards(tag + " candidates", dc);
    failures += verify_guards(tag + " unary", du);
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
    failures += tie_boundary_case(4001U);
    for (const std::int32_t width : {8, 16}) {
        failures += candidates_case(4992, width, 2001U + static_cast<std::uint32_t>(width));
        failures += lattice_case(kDFlash2SelectorRank, 5120, width, 1,
                                 3001U + static_cast<std::uint32_t>(width));
    }
    failures += lattice_case(kDFlash2SelectorRank, 5120, 8, 3, 3083U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash2_selector\n";
    return failures == 0 ? 0 : 1;
}