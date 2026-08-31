# NPChallenge · NumPy 挑战 🧠

> 用精简 NumPy 实现注意力与卷积，并通过小型基准进行学习和比较。

[English](README.md) · [简体中文](README.zh-CN.md)

## 快速开始

```bash
python3 attention.py
python3 conv2d.py
python3 conv_bench.py
```

项目探索一个实践问题：实现核心神经网络算子时，清晰且少依赖的 NumPy 代码能走多远？它优先考虑**可读和可检查**，不追求框架完整性或 GPU 性能。

文件包括注意力、二维卷积及基准实验。结果取决于硬件和 NumPy 版本；这是教育性挑战，不是高性能深度学习库的替代品。当前版本为 **0.1.0**（[`VERSION`](VERSION)），变更见 [`CHANGELOG.md`](CHANGELOG.md)。欢迎在保持参考行为的同时提交带正确性检查或小型基准的改进。Agent 可以协助分诊、调查、测试和文档工作；维护者负责审查和合并。详见 [`CONTRIBUTING.md`](CONTRIBUTING.md) 和 [`docs/project-status.md`](docs/project-status.md)。许可证：AGPL-3.0-only，见 [`LICENSE`](LICENSE)。
