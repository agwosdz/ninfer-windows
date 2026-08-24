# NInfer compressed-KV E8 codec tests

This directory holds the correctness oracle + microbenchmark for the
compressed-KV cache codecs powering the `rk2v4-e8` and `rk4v4-e8` live
KV-quantization modes (and the `rk8v4` / `rk4v4` rotated/packed modes they
share machinery with).

It is fully standalone: the kernels are self-contained CUDA (no dependency
on `ninfer_core` or any model artifact).

## What is verified

`verify_1m_retrieval.cu` synthesizes 1,000,000 realistic transformer
key tokens (256-dim, Gaussian), embeds **five high-magnitude "needles"** at
5% / 25% / 50% / 75% / 95% offsets, and asks both codecs to reconstruct
attention scores against an exact FP32 ground-truth dot product:

- **Method 1 — Two-Stage E8 Root Codec** (`rk2v4-e8`): 2 bits/dim,
  208 B/token/head K+V incl. group scales (~4.9x vs the 1024 B bf16 K+V).
- **Method 2 — Exact Conway-Sloane E8 Lattice Point** (`rk4v4-e8`):
  4-bit code + coset bit per dim (doubled int8 code c = 2·p in [-15, 15]),
  400 B/token/head K+V incl. group scales (~2.6x vs bf16).

Metrics reported per method: cosine similarity vs FP32 reference, mean abs
error, max abs error, and per-needle retrieval ranking.

## Verified result (GeForce RTX 5090, sm_120a)

- **Needle retrieval: 100%** — all 5 embedded needles correctly recovered
  at their exact token indices by both the 240-root and general-lattice
  codecs (needle scores `> 15.0`, matching the FP32 reference which is also
  `> 15.0`).
- **Cosine similarity ≈ 100%** between the quantized attention output and
  the FP32 reference across the full 1M-token corpus.
- `test_e8_lattice_oracle.cu` separately checks the **production**
  `rk4v4-e8` codec math in `src/ops/kernel/e8_lattice.cuh`: the device warp
  projection must equal the exact nearest E8 lattice point (independent FP64
  algebraic reference cross-validated by exhaustive enumeration), the doubled
  codes must be exact in [-15, 15] with parity = coset, the exact
  reconstruction must match plain `rk4v4`'s quality within 1% (parity is the
  expected relationship: same step grid, E8 shape at 1/16 density), and it
  must beat the superseded half-coset approximation.
- The same `rk2v4-e8` path is reported by the source fork as having served a
  Qwen3.6-27B NVFP4 model end-to-end at **262,144-token context** on a
  32 GB-class Blackwell GPU (sm_120a); that run has not been independently
  reproduced in this tree.

This is exactly the property that makes 262K context fit a 32 GB GPU: the
KV cache lives on-die at 2–5 bits per dimension instead of Full-Precision.

## Building & running

From the NInfer top level (after CUDA 13.1+ / Blackwell is configured):

```sh
cmake -S . -B build -GNinja -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -R ninfer_kv_e8 --output-on-failure
```

Standalone (no CMake needed):

```sh
nvcc -arch=sm_120a -O3 tools/test_kv/test_e8_codec.cu \
     tools/test_kv/verify_1m_retrieval.cu -o verify_1m_retrieval
./verify_1m_retrieval            # default: 1,000,000 tokens
./verify_1m_retrieval 2000000    # optional: corpus size
```

## Attribution

The compressed-KV cache design — Hadamard-rotated K/V, int4-packed V,
and the **E8 Conway-Sloane lattice / 240-root codec mathematics** that make
runnable 2-bit and 4-bit KV caches possible — originates from
**UDPSendToFailed/ninfer-4090**, a fork of **Neroued/ninfer** targeting the
NVIDIA Ada (sm_89 / GeForce RTX 4090) generation, itself derived in turn
from **Don-Chad/ninfer-3090**.

**Full credit for the codec mathematics and the collapsed-KV cache
architecture goes to the ninfer-4090 fork author.** This tree ports that
work onto the official Blackwell (sm_120a) upstream codebase for
live-KV-quantization at long context on 32 GB-class GPUs. One deviation
from the source of truth: the ported `rk4v4-e8` projection discarded the
D8 + 1/2 coset (half-coset approximation); it has been replaced here by
exact coset-carrying doubled codes so the production path stores the true
nearest E8 point (covered by the production-codec oracle above).