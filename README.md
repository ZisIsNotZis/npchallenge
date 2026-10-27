[English](README.md) · [简体中文](README.zh-CN.md)

<p align="left"><img src="assets/logo.svg" alt="npchallenge" width="220" height="44"></p>

# NPChallenge

Pure-NumPy causal attention and 2D convolution, small enough to read in one sitting.

**Two self-contained operators — multi-head causal attention and NCHW convolution — that pick the fastest path NumPy can give them, with every trick written out in readable code instead of hidden inside a framework.**

A teaching artifact, not a library: there is nothing to `pip install`, no build step, and no abstraction between you and `np.matmul`. The question it answers is how far careful, dependency-light NumPy gets when you implement the operators a transformer or a CNN actually needs — and where it stops.

## Quick start

```bash
git clone https://github.com/ZisIsNotZis/npchallenge.git
cd npchallenge
python3 -m pip install numpy
```

Then, from the repo root (the scripts import each other by filename, so no packaging step exists):

```python
import numpy as np
from attention import causal_attention
from conv2d import conv2d

Q = K = V = np.random.randn(2, 4, 256, 64).astype(np.float32)
print(causal_attention(Q, K, V).shape)   # (2, 4, 256, 64)

x = np.random.randn(1, 3, 32, 32).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
print(conv2d(x, w, pad=1).shape)         # (1, 16, 32, 32)
```

Run the benchmarks (they also verify correctness against naive references):

```bash
python3 bench_final.py                    # conv2d vs. 4 baselines, timings + GFLOPS
OPENBLAS_NUM_THREADS=6 python3 conv_bench.py   # 8 conv variants x 10 shapes
```

## What it can do now

- **`causal_attention(Q, K, V, scale=None, mask=None, block_size=64)`** — multi-head causal attention on `(batch, heads, seq, head_dim)` tensors, with an optional extra additive mask (padding masks work unchanged).
- **Causality exploited, not just masked** — below `S = 128` it does the plain full `S × S` product; at or above it tiles the sequence and computes only the lower-triangular part, roughly halving the matmul work.
- **`conv2d(x, w, pad=0, stride=1, dilation=1)`** — NCHW input, OIHW kernel, with padding, stride and dilation given as ints or `(h, w)` pairs; output geometry is computed with the standard formula.
- **Four convolution strategies chosen at call time** — `1×1` einsum, im2col + matmul for small outputs, a strided-view einsum for large ones, and a minimal-pad FFT path for big kernels (`KH·KW ≥ 512`).
- **In-place, numerically stable softmax** — max subtraction, `exp` and divide all write back into the score buffer; masks are cached per `(shape, dtype)`.
- **Benchmark harnesses that check before they time** — `conv_bench.py` compares eight implementations against a naive loop convolution and prints which one wins per shape; `bench_final.py` reports GFLOPS for `conv2d`.
- **One runtime dependency** — NumPy, nothing else.

## How it works

Two flat modules, each a single public function over private kernels; the benchmark scripts sit beside them and import them directly.

```
npchallenge/
├── attention.py          # causal_attention: causal-mask cache, in-place softmax,
│                         # full vs. blocked dispatch
├── conv2d.py             # conv2d: output geometry, stride-tricks view,
│                         # 1×1 / im2col / einsum / FFT dispatch
├── conv_bench.py         # 8 conv implementations benchmarked against each other
├── bench_final.py        # conv2d vs. naive + 3 baselines, with GFLOPS
├── docs/project-status.md
├── assets/logo.svg
├── CHANGELOG.md · VERSION · CONTRIBUTING.md · LICENSE
└── README.md · README.zh-CN.md
```

**Attention.** `causal_attention` normalizes `scale` to `1/√D`, then dispatches. With a user mask or short sequences it builds the full score matrix, adds the mask, adds a cached `-inf` upper-triangular mask, runs softmax in place and multiplies by `V`. For long sequences it walks query blocks of `block_size`: each block's scores are computed against the keys up to its end, masked with a triangle offset by the block start, softmaxed and multiplied by the matching `V` slice, writing straight into the preallocated output.

**Convolution.** `conv2d` pads once, builds a zero-copy `as_strided` view of shape `(N, C, H_out, W_out, KH, KW)`, and then decides how to contract it: `1×1` kernels avoid the strided view entirely and use a plain channel einsum; views under ~196k elements go through a reshape to im2col and a single matmul; larger views stay as an einsum so NumPy's C kernel handles the strides without copying; kernels with `KH·KW ≥ 512` under unit stride and dilation drop to FFT with minimal padding, guarded by a `C·C_out ≤ 8192` cap so the frequency-domain multiply cannot blow up.

The known limits are deliberate: everything is float32/float64 CPU work, timings are BLAS- and core-count-dependent, and the recorded numbers in the module docstrings were measured on one machine (i7-6850K, OpenBLAS, AVX2) — treat them as a baseline to beat, not a promise.

## Development

There is no build, no package and no test runner. The correctness checks live inside the benchmark scripts: `conv_bench.py` validates every implementation against `conv_naive` on a tiny input before timing, and `bench_final.py` checks `conv2d` against `baseline_naive` for padded, dilated and strided cases.

```bash
python3 bench_final.py
OPENBLAS_NUM_THREADS=6 python3 conv_bench.py
```

For comparable numbers, pin the thread count to your physical cores (the scripts default to 6) and report CPU, NumPy version and BLAS in any benchmark claim.

## Contributing

Small, focused patches are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The bar is that the reference behavior stays simple and every performance claim comes with a reproducible measurement.

## Roadmap

**Implemented**

- Causal multi-head attention with full and blocked kernels.
- 2D convolution with 1×1, im2col, einsum and FFT paths, plus pad/stride/dilation.
- Two benchmark harnesses with built-in correctness verification.
- Versioned documentation: [`CHANGELOG.md`](CHANGELOG.md), [`VERSION`](VERSION), [`docs/project-status.md`](docs/project-status.md).

**Planned**

- Nothing scheduled. This is a closed milestone (2026-09-29): the teaching implementations are complete and no further development is planned unless the project's inputs or goals change.

The obvious directions, if it were reopened, are the ones the code already points at — a backward pass, a batched over-`head_dim` variant, and turning the ad-hoc verification in the benchmark scripts into a real test file. None of them exist today.

## License

AGPL-3.0-only. See [`LICENSE`](LICENSE). Copyright (C) 2026 ZisIsNotZis.
