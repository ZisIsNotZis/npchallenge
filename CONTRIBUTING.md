# Contributing

Thanks for looking. This repository is a small teaching artifact, so the bar is quality, not volume.

## Scope

In: clearer kernels, correctness fixes, new dispatch strategies with evidence, better benchmark coverage, documentation. Out: turning it into a framework — no packaging, no new dependencies beyond NumPy, no multi-file abstraction layer.

## Running the checks

There is no test runner. Benchmarks verify correctness before they time:

```bash
python3 bench_final.py                          # conv2d vs. naive + 3 baselines
OPENBLAS_NUM_THREADS=6 python3 conv_bench.py    # 8 conv variants, correctness + timings
```

For attention, a quick self-check against a plain masked-matmul reference on a small `(B, H, S, D)` tensor is enough — include it in the PR description if the change touches the blocked path.

## Pull requests

- One logical change per PR, with the reason in the title (`perf: avoid the strided copy for 5×5 kernels`).
- Any performance claim needs the measurement: CPU, NumPy version, BLAS, thread count, and the command you ran. Numbers recorded in a docstring must say which machine produced them.
- Behavior changes must keep the public signatures of `causal_attention` and `conv2d` stable, or explain why they cannot.
- Reference behavior stays readable: if a fast path needs a comment to be understood, add the comment.

## Issues

Include the script, the command, the full traceback, and your environment (`python3 -c "import numpy; print(numpy.__version__)"`, OS, CPU). A failing input with expected vs. actual output is worth more than a description.

Agents may triage, investigate, test, and document accepted work; maintainers review and merge.
