# KV-dtype correctness diagnosis (rotated/packed line)

**Status: open investigation — not a resolved bug.**

Measured on `models/qwen3_8_27b_dflash2.ninfer` (qwen3.8-27b, greedy, mtp0):

| kv-dtype | output | verdict |
|---|---|---|
| bf16 | coherent, correct | clean |
| int8 | coherent, correct | clean |
| rk8v4 | run-together garbage | broken |
| rk4v4 | `bm_ai_unk g (dfn\...` soup | broken |
| rk4v4-e8 | token soup (`<⁍ x_movedq two x...`) | broken |
| rk2v4-e8 | coherent but wrong decomposition | broken-with-corruption |

Repro (each starts a fresh server):

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python tools/bench/_debug_det_kvtype.py     # determinism + content per kv
```

Key facts established:

1. **Deterministic.** The same request through `rk4v4-e8` twice returns the
   identical garbage (not RNG, not a transient).
2. **Rotation math is correct.** `gqa_kv_hadamard64`
   (`src/ops/kernel/gqa_attention_kv_quant.cuh`) is self-inverse and
   norm-preserving on a numerical 64-dim deep check.
3. **The E8 codec itself is unit-tested** (`tools/test_kv/`), so the pure
   lattice encode/decode primitives are not the suspect.
4. Therefore the fault is in the **fused packed-KV integration**: the int4
   nibble write/read indexing (`gqa_kv_i4_code_index`, `d/2`, lane-pair
   `shfl_down` packing) or the per-group scale round-trip between the
   write-time `amax/7` (or `/14` for E8) and read-time `v_scale` fetch in
   `gqa_attention_decode_i8.cuh` (lines ~639-643).

Next step (not yet done): a packed-KV round-trip oracle test under
`tests/ops/` — feed a known 256-dim vector per group, write via the
encode path, read via the decode path stages, and compare; that isolates the
layout/scale misuse without the whole model.

Caveat: `rk2v4-e8`'s "least broken" behavior (coherent sentences, wrong
numbers) is consistent with a scale error rather than pure index garbage —
the E8 K unit-tests pass, so K is likely fine and the corruption may be
**packed-V only** (plain int4 V), which every rk variant shares (`PackedV`).
A focused test should therefore check **V alone** first.