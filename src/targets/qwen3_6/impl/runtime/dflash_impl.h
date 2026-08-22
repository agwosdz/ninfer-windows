#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"
#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_predecessor_ids.h"
#include "ninfer/ops/dflash2_selector_lattice.h"
#include "ninfer/ops/residual_add.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash && !V::DFlashConfig::is_v2) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            Tensor key;
            Tensor value;
            if constexpr (Config::is_v2) {
                auto attn_roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, layer_columns);
                ops::rmsnorm(layer_context, weight.input_norm, Config::rms_epsilon, false,
                             attn_roots.hidden, state.execution.device.stream);
                Tensor attn_dynamic = state.execution.work.alloc(
                    DType::BF16, {Config::conv_projection_rows, layer_columns});
                ops::linear(attn_roots.hidden, weight.attention_conv_projection, attn_dynamic,
                            state.execution.device.stream);
                Tensor noise_conv = state.execution.work.alloc(
                    DType::BF16, {Config::hidden, layer_columns});
                ops::dflash2_dynamic_conv(attn_roots.hidden, attn_dynamic,
                                          weight.attention_conv_base, 0, layer_width, noise_conv,
                                          state.execution.device.stream);
                Tensor query_raw =
                    attn_roots.query_raw.view({Config::head_dim, Config::query_heads, layer_columns});
                Tensor key_raw =
                    attn_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
                value = attn_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
                // QKV: plain W8 linear into a packed 6144-row buffer (the drafter
                // QKV shape is 35B-incompatible for the fused attn_input_proj op),
                // then scatter rows into the existing q/k/v recipe buffers.
                Tensor qkv_packed = state.execution.work.alloc(
                    DType::BF16, {Config::query_size + 2 * Config::kv_size, layer_columns});
                ops::linear(noise_conv, weight.query_key_value, qkv_packed,
                            state.execution.device.stream);
                {
                    const std::size_t row_bytes =
                        static_cast<std::size_t>(layer_columns) * dtype_size(DType::BF16);
                    CUDA_CHECK(cudaMemcpyAsync(query_raw.data, qkv_packed.data,
                                               static_cast<std::size_t>(Config::query_size) * row_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               state.execution.device.stream));
                    CUDA_CHECK(cudaMemcpyAsync(key_raw.data,
                                               static_cast<char*>(qkv_packed.data) +
                                                   static_cast<std::size_t>(Config::query_size) * row_bytes,
                                               static_cast<std::size_t>(Config::kv_size) * row_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               state.execution.device.stream));
                    CUDA_CHECK(cudaMemcpyAsync(value.data,
                                               static_cast<char*>(qkv_packed.data) +
                                                   static_cast<std::size_t>(Config::query_size + Config::kv_size) * row_bytes,
                                               static_cast<std::size_t>(Config::kv_size) * row_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               state.execution.device.stream));
                }
                key = attn_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                          key, state.execution.device.stream);
            } else {
                auto layer_roots =
                    workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
                Tensor key_raw =
                    layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
                value =
                    layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
                Tensor value_flat = value.view({Config::kv_size, layer_columns});
                ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                                 value_flat, state.execution.device.stream);
                key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                          key, state.execution.device.stream);
            }
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.stream);
            }
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                             Config::attention_scale, Config::local_capacity,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                        state.execution.device.stream);
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}



template <class V, class Context>
void propose_batch_v2_impl(Context& state, qwen3_6::DFlashDecodeState& frame,
                           std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::DFlashConfig::is_v2) {
        throw std::logic_error("DFlash2 proposal is unavailable for this target");
    } else {
        using Config                = typename V::DFlashConfig;
        const std::int32_t width    = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns  = width * batch_size;
        Tensor anchors              = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers            = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns        = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes                = frame.lanes.slice(0, 0, batch_size);
        Tensor ids                  = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions            = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts               = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);

        // inpL: the block input embeddings — the FIXED attention base for all
        // layers (block-diffusion: no residual stream between layers).
        Tensor inpL = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, inpL,
                       state.execution.device.stream);

        // residual: the running state, initialized to inpL. Updated at the end
        // of each layer (mlp_out_conv + ffn_inp); used for the final layer's
        // output.
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        const std::size_t hidden_bytes =
            static_cast<std::size_t>(Config::hidden) * columns * dtype_size(DType::BF16);
        CUDA_CHECK(cudaMemcpyAsync(residual.data, inpL.data, hidden_bytes,
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));

            // === Attention sublayer ===
            auto attn_roots = workspace_recipe::dflash_attention<Config>(state.execution.work, columns);

            // noise_norm = rmsnorm(inpL, input_norm) — norm of the FIXED base
            ops::rmsnorm(inpL, weight.input_norm, Config::rms_epsilon, false, attn_roots.hidden,
                         state.execution.device.stream);

            // attn_dynamic = linear(noise_norm, attn_conv_proj)
            Tensor attn_dynamic = state.execution.work.alloc(
                DType::BF16, {Config::conv_projection_rows, columns});
            ops::linear(attn_roots.hidden, weight.attention_conv_projection, attn_dynamic,
                        state.execution.device.stream);

            // noise_conv = two-tap dynamic conv(side=0) of noise_norm
            Tensor noise_conv = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_dynamic_conv(attn_roots.hidden, attn_dynamic, weight.attention_conv_base,
                                      0, Config::block_size, noise_conv,
                                      state.execution.device.stream);

            // QKV projection from the conv'd signal
            Tensor query_raw = attn_roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
            Tensor key_raw   = attn_roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
            Tensor value     = attn_roots.value.view({Config::head_dim, Config::kv_heads, columns});
            // QKV: plain W8 linear into a packed 6144-row buffer (the drafter
            // QKV shape is 35B-incompatible for the fused attn_input_proj op),
            // then scatter rows into the existing q/k/v recipe buffers.
            Tensor qkv_packed = state.execution.work.alloc(
                DType::BF16, {Config::query_size + 2 * Config::kv_size, columns});
            ops::linear(noise_conv, weight.query_key_value, qkv_packed,
                        state.execution.device.stream);
            {
                const std::size_t row_bytes =
                    static_cast<std::size_t>(columns) * dtype_size(DType::BF16);
                CUDA_CHECK(cudaMemcpyAsync(query_raw.data, qkv_packed.data,
                                           static_cast<std::size_t>(Config::query_size) * row_bytes,
                                           cudaMemcpyDeviceToDevice, state.execution.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(key_raw.data,
                                           static_cast<char*>(qkv_packed.data) +
                                               static_cast<std::size_t>(Config::query_size) * row_bytes,
                                           static_cast<std::size_t>(Config::kv_size) * row_bytes,
                                           cudaMemcpyDeviceToDevice, state.execution.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(value.data,
                                           static_cast<char*>(qkv_packed.data) +
                                               static_cast<std::size_t>(Config::query_size + Config::kv_size) * row_bytes,
                                           static_cast<std::size_t>(Config::kv_size) * row_bytes,
                                           cudaMemcpyDeviceToDevice, state.execution.device.stream));
            }

            // Head norms + RoPE
            Tensor query = attn_roots.query.view({Config::head_dim, Config::query_heads, columns});
            Tensor key   = attn_roots.key.view({Config::head_dim, Config::kv_heads, columns});
            ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                         state.execution.device.stream);
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query, key,
                      state.execution.device.stream);

            // Batch views for the SWA op
            Tensor query_batch =
                query.view({Config::head_dim, Config::query_heads, width, batch_size});
            Tensor key_batch =
                key.view({Config::head_dim, Config::kv_heads, width, batch_size});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, width, batch_size});
            Tensor attention_batch = attn_roots.attention.view(
                {Config::head_dim, Config::query_heads, width, batch_size});

            // Non-causal sliding-window attention (all v2 layers are local)
            ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                     Config::attention_scale, Config::local_capacity,
                     dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                     envelopes.local, state.execution.work, attention_batch,
                     state.execution.device.stream);

            // attn_out = linear(attention, attention_output)
            Tensor attn_out = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::linear(attn_roots.attention.view({Config::query_size, columns}),
                        weight.attention_output, attn_out, state.execution.device.stream);

            // attn_out_conv = two-tap dynamic conv(side=1) of attn_out
            Tensor attn_out_conv = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_dynamic_conv(attn_out, attn_dynamic, weight.attention_conv_base,
                                      1, Config::block_size, attn_out_conv,
                                      state.execution.device.stream);

            // ffn_inp = attn_out_conv + inpL
            Tensor ffn_inp = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            CUDA_CHECK(cudaMemcpyAsync(ffn_inp.data, attn_out_conv.data, hidden_bytes,
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
            ops::residual_add(inpL, ffn_inp, state.execution.device.stream);

            // === FFN sublayer ===
            auto mlp_roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);

            // ffn_norm = rmsnorm(ffn_inp, post_attention_norm)
            ops::rmsnorm(ffn_inp, weight.post_attention_norm, Config::rms_epsilon, false,
                         mlp_roots.hidden, state.execution.device.stream);

            // ffn_dynamic = linear(ffn_norm, mlp_conv_proj)
            Tensor ffn_dynamic = state.execution.work.alloc(
                DType::BF16, {Config::conv_projection_rows, columns});
            ops::linear(mlp_roots.hidden, weight.mlp_conv_projection, ffn_dynamic,
                        state.execution.device.stream);

            // ffn_conv = two-tap dynamic conv(side=0) of ffn_norm
            Tensor ffn_conv = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_dynamic_conv(mlp_roots.hidden, ffn_dynamic, weight.mlp_conv_base,
                                      0, Config::block_size, ffn_conv,
                                      state.execution.device.stream);

            // SwiGLU: intermediate = silu(gate) * up. The W8 linear_swiglu op only has
            // kernels for the 35B drafter shape, so the v2 drafter decomposes the
            // projection like the MTP head: W8 linear gate/up + elementwise silu_mul.
            Tensor ffn_gate_up = state.execution.work.alloc(
                DType::BF16, {2 * Config::intermediate, columns});
            ops::linear(ffn_conv, weight.gate_up, ffn_gate_up, state.execution.device.stream);
            ops::silu_mul(ffn_gate_up.slice(0, 0, Config::intermediate),
                          ffn_gate_up.slice(0, Config::intermediate, Config::intermediate),
                          mlp_roots.intermediate, state.execution.device.stream);

            // mlp_out = linear(intermediate, down)
            Tensor mlp_out = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::linear(mlp_roots.intermediate, weight.down, mlp_out,
                        state.execution.device.stream);

            // mlp_out_conv = two-tap dynamic conv(side=1) of mlp_out
            Tensor mlp_out_conv = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_dynamic_conv(mlp_out, ffn_dynamic, weight.mlp_conv_base,
                                      1, Config::block_size, mlp_out_conv,
                                      state.execution.device.stream);

            // residual = mlp_out_conv + ffn_inp
            CUDA_CHECK(cudaMemcpyAsync(residual.data, mlp_out_conv.data, hidden_bytes,
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
            ops::residual_add(ffn_inp, residual, state.execution.device.stream);
        }

        // After all layers: final_norm → target output head → BF16 logits over
        // the full block (including the anchor token at position 0).
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, columns});
        ops::rmsnorm(residual, state.execution.model.dflash->final_norm, Config::rms_epsilon,
                     false, proposal_hidden, state.execution.device.stream);

        Tensor logits = state.execution.work.alloc(
            DType::BF16, {TextConfig::output_rows, columns});
        ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                    state.execution.device.stream);

        // Select top-16 candidates per draft position
        Tensor candidates = state.execution.work.alloc(
            DType::I32, {Config::selector_top_k, columns});
        Tensor unary = state.execution.work.alloc(
            DType::FP32, {Config::selector_top_k, columns});
        ops::dflash2_select_candidates(logits, candidates, unary,
                                       state.execution.device.stream);

        // hidden_pos = linear(inpL, selector_hidden_proj) — the rank-256
        // projection of the draft input embeddings
        Tensor hidden_pos = state.execution.work.alloc(
            DType::BF16, {Config::selector_rank, columns});
        ops::linear(inpL, state.execution.model.dflash->selector_hidden_projection, hidden_pos,
                    state.execution.device.stream);

        // successor = embedding(candidates_flat, successor_codebook)
        const std::int32_t top_k_cols = Config::selector_top_k * columns;
        Tensor candidates_flat = candidates.view({top_k_cols});
        Tensor successor = state.execution.work.alloc(
            DType::BF16, {Config::selector_rank, top_k_cols});
        ops::embedding(candidates_flat,
                       state.execution.model.dflash->selector_successor_codebook, successor,
                       state.execution.device.stream);

        // predecessor_ids = block-shifted candidates (anchor broadcast at pos 0/1)
        Tensor predecessor_ids = state.execution.work.alloc(
            DType::I32, {Config::selector_top_k, columns});
        ops::dflash2_predecessor_ids(candidates, anchors, Config::block_size, predecessor_ids,
                                     state.execution.device.stream);
        Tensor predecessor_ids_flat = predecessor_ids.view({top_k_cols});
        Tensor predecessor = state.execution.work.alloc(
            DType::BF16, {Config::selector_rank, top_k_cols});
        ops::embedding(predecessor_ids_flat,
                       state.execution.model.dflash->selector_predecessor_codebook, predecessor,
                       state.execution.device.stream);

        // The lattice op expects [rank, top_k, T] (T = columns); the gather wrote a
        // flat [rank, top_k*T] buffer with the identical dim0-fastest memory layout
        // (element (s,t) at flat column s + top_k*t), so view it to the 3D shape.
        Tensor successor_3d  = successor.view({Config::selector_rank, Config::selector_top_k, columns});
        Tensor predecessor_3d = predecessor.view({Config::selector_rank, Config::selector_top_k, columns});

        // Lattice: [hidden, columns] f32 rows
        // [top_k ids | top_k×top_k scores | zero-pad to hidden]
        Tensor lattice = state.execution.work.alloc(DType::FP32, {Config::hidden, columns});
        ops::dflash2_selector_lattice(hidden_pos, successor_3d, predecessor_3d, candidates, unary,
                                      Config::hidden, Config::block_size, lattice,
                                      state.execution.device.stream);

        // D2H + host greedy path-trace (eager; cannot be graph-captured).
        // pred = 0 (anchor row); per position read the 16 successor scores
        // for the current predecessor, argmax → next draft token.
        {
            const std::size_t lattice_bytes =
                static_cast<std::size_t>(Config::hidden) * columns * sizeof(float);
            auto* host_lattice = static_cast<float*>(std::malloc(lattice_bytes));
            if (host_lattice == nullptr) {
                throw std::bad_alloc();
            }
            CUDA_CHECK(cudaMemcpyAsync(host_lattice, lattice.data, lattice_bytes,
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));

            const std::int32_t top_k   = Config::selector_top_k;
            const std::int32_t packed  = Config::hidden;
            const std::int32_t k_drafts = static_cast<std::int32_t>(k);

            // host_drafts[pos, batch] — the traced draft token ids
            auto* host_drafts = static_cast<std::int32_t*>(
                std::malloc(static_cast<std::size_t>(k_drafts) * batch_size *
                            sizeof(std::int32_t)));
            if (host_drafts == nullptr) {
                std::free(host_lattice);
                throw std::bad_alloc();
            }

            for (std::int32_t b = 0; b < batch_size; ++b) {
                std::int32_t pred = 0;  // anchor predecessor
                for (std::int32_t pos = 1; pos <= k_drafts; ++pos) {
                    // Lattice columns follow the [width, batch] draft-block layout
                    // (dim0 fastest): flat column for block position `pos` of batch
                    // `b` is pos + b*width.
                    const std::size_t col  = static_cast<std::size_t>(pos) +
                                             static_cast<std::size_t>(b) * static_cast<std::size_t>(width);
                    const float* row       = host_lattice + col * packed;
                    const float* scores    = row + top_k + static_cast<std::size_t>(pred) * top_k;
                    std::int32_t best_s    = 0;
                    float best_score       = scores[0];
                    for (std::int32_t s = 1; s < top_k; ++s) {
                        if (scores[s] > best_score) {
                            best_score = scores[s];
                            best_s     = s;
                        }
                    }
                    // The traced token id is the best_s-th candidate at this
                    // position (row[0..top_k-1] are the candidate ids as f32).
                    // draft_tokens is [k, batch] with dim0 fastest, so the flat
                    // index for (draft p = pos-1, batch b) is p + b*k.
                    host_drafts[b * k_drafts + (pos - 1)] =
                        static_cast<std::int32_t>(row[best_s]);
                    pred = best_s;
                }
            }
            std::free(host_lattice);

            // Write traced drafts to the device tensor the verify/accept reads.
            CUDA_CHECK(cudaMemcpyAsync(drafts.data, host_drafts,
                                       static_cast<std::size_t>(k_drafts) * batch_size *
                                           sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, state.execution.device.stream));
            std::free(host_drafts);
        }

        state.execution.work.reset();
    }
}


auto dflash_decode_batch_body_v2(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                                 DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash2 decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        propose_batch_v2_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}
} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    // The scheduler reaches the v1 entry points for every DFlash target.
    // V2 variants (dflash2 block-diffusion drafter) run the v2 body: its
    // attention/FFN differ and the v1 path uses 35B-only fused ops
    // (attn_input_proj / linear_swiglu).
    if constexpr (Variant::DFlashConfig::is_v2) {
        auto body = dflash_decode_batch_body_v2(state, batch_size, k, envelopes,
                                                 target_envelope);
        capture_graph(state, definition, body);
    } else {
        auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
        capture_graph(state, definition, body);
    }
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    if constexpr (Variant::DFlashConfig::is_v2) {
        auto body = dflash_decode_batch_body_v2(state, batch_size, k, envelopes,
                                                 target_envelope);
        run_prepared(state, executable, body);
    } else {
        auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
        run_prepared(state, executable, body);
    }
}


void capture_dflash_decode_batch_v2(DFlashBatchContext& state, std::int32_t batch_size,
                                    std::uint32_t k, DFlashEnvelopes envelopes,
                                    ops::GqaExecutionEnvelope target_envelope,
                                    DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body_v2(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch_v2(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                            DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                            DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body_v2(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
