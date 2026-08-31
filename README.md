# NPChallenge 🧠

> Minimal NumPy implementations of attention and convolution, with small benchmarks for learning and comparison.

[English](README.md) · [简体中文](README.zh-CN.md)

## Quickstart

```bash
python3 attention.py
python3 conv2d.py
python3 conv_bench.py
```

The project asks a practical question: how far can clear, dependency-light NumPy code go when implementing core neural-network operators? It favors **readability and inspectability** over framework completeness or GPU performance.

## Contents

- `attention.py` — attention implementation;
- `conv2d.py` — 2D convolution implementation;
- `conv_bench.py`, `bench_final.py` — benchmark experiments.

Results depend on hardware and NumPy version. This is an educational challenge, not a replacement for optimized deep-learning libraries. Contributions should preserve simple reference behavior and include a small benchmark or correctness check. Licensed under AGPL-3.0-only; see [`LICENSE`](LICENSE).

## Versioning and help

Current version: **0.1.0** ([`VERSION`](VERSION)); changes are in [`CHANGELOG.md`](CHANGELOG.md). Open focused issues or PRs with the command, environment, and evidence. Agents may help triage, investigate, test, and document accepted work; maintainers review and merge. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/project-status.md`](docs/project-status.md).
