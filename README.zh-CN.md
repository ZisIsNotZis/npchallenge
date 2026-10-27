[English](README.md) · **简体中文**

<p align="left"><img src="assets/logo.svg" alt="npchallenge" width="220" height="44"></p>

# NPChallenge

纯 NumPy 的因果注意力与二维卷积，短到可以一口气读完。

**两个自成一体、可直接阅读的算子——多头因果注意力与 NCHW 卷积——在 NumPy 允许的范围内自动挑选最快的实现路径，每一种技巧都摊开写在代码里，而不是藏在框架内部。**

这是教学产物，不是库：无需 `pip install`，没有构建步骤，你和 `np.matmul` 之间也没有任何抽象层。它想回答的问题是：亲手实现 Transformer 和 CNN 真正需要的算子时，少依赖、写得清楚的 NumPy 到底能走多远，又在哪里到头。

## 快速开始

```bash
git clone https://github.com/ZisIsNotZis/npchallenge.git
cd npchallenge
python3 -m pip install numpy
```

然后在仓库根目录下（脚本按文件名互相导入，没有打包步骤）：

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

运行基准测试（它们会先用朴素实现校验正确性，再计时）：

```bash
python3 bench_final.py                         # conv2d 对比 4 个基线，给出耗时与 GFLOPS
OPENBLAS_NUM_THREADS=6 python3 conv_bench.py   # 8 种卷积实现 × 10 组形状
```

## 目前能做什么

- **`causal_attention(Q, K, V, scale=None, mask=None, block_size=64)`**——在 `(batch, heads, seq, head_dim)` 张量上做多头因果注意力，支持额外的加性掩码（padding mask 可直接使用）。
- **不仅遮住因果性，还利用它**——`S = 128` 以下走完整的 `S × S` 乘积；达到或超过则分块处理，只算下三角部分，大致省掉一半矩阵乘工作量。
- **`conv2d(x, w, pad=0, stride=1, dilation=1)`**——NCHW 输入、OIHW 卷积核，padding、stride、dilation 均可给整数或 `(h, w)` 对；输出尺寸按标准公式计算。
- **调用时四选一的卷积策略**——`1×1` 用 einsum，小输出用 im2col + 矩阵乘，大输出用跨步视图上的 einsum，大卷积核（`KH·KW ≥ 512`）走最小填充的 FFT。
- **原地、数值稳定的 softmax**——减最大值、`exp`、相除全部写回同一块分数缓冲区；掩码按 `(shape, dtype)` 缓存。
- **先校验再计时的基准脚本**——`conv_bench.py` 把八种实现与朴素循环卷积对比，并打印每种形状下的胜者；`bench_final.py` 报告 `conv2d` 的 GFLOPS。
- **唯一运行时依赖是 NumPy**。

## 实现思路

两个扁平模块，每个都是「一个公开函数 + 若干私有内核」；基准脚本与之并列，直接按文件名导入。

```
npchallenge/
├── attention.py          # causal_attention：掩码缓存、原地 softmax、
│                         # 完整 / 分块两种路径的分发
├── conv2d.py             # conv2d：输出尺寸计算、跨步视图、
│                         # 1×1 / im2col / einsum / FFT 分发
├── conv_bench.py         # 8 种卷积实现互相基准对比
├── bench_final.py        # conv2d 对比朴素实现与 3 个基线，附 GFLOPS
├── docs/project-status.md
├── assets/logo.svg
├── CHANGELOG.md · VERSION · CONTRIBUTING.md · LICENSE
└── README.md · README.zh-CN.md
```

**注意力。** `causal_attention` 先把 `scale` 归一为 `1/√D`，再分发。带用户掩码或序列较短时，它构造完整分数矩阵，加上用户掩码，再加上按 `(shape, dtype)` 缓存的 `-inf` 上三角掩码，原地 softmax 后乘以 `V`。序列较长时，它按 `block_size` 逐块遍历查询：每块的分数只与该块末尾之前的键相乘，用按块起点偏移的三角掩码遮罩，softmax 后乘以对应的 `V` 切片，直接写入预分配的输出。

**卷积。** `conv2d` 先做一次填充，再用 `as_strided` 建出形状为 `(N, C, H_out, W_out, KH, KW)` 的零拷贝视图，然后决定如何收缩它：`1×1` 卷积核完全绕开跨步视图，直接做通道 einsum；视图元素数在约 19.6 万以下时，reshape 成 im2col 后一次矩阵乘；更大的视图保持 einsum，让 NumPy 的 C 内核处理跨步、避免拷贝；在 stride 与 dilation 均为 1 的前提下，`KH·KW ≥ 512` 的核改用最小填充的 FFT，并以 `C·C_out ≤ 8192` 为上限，避免频域乘法膨胀。

它的边界是刻意保留的：一切都是 CPU 上的 float32/float64 计算，耗时取决于 BLAS 与核心数，模块文档字符串里的数字只在同一台机器上测得（i7-6850K、OpenBLAS、AVX2）——请把它们当作待超越的基线，而不是承诺。

## 开发

没有构建、没有打包、也没有测试框架。正确性校验写在基准脚本里：`conv_bench.py` 在计时前用极小输入把每种实现与 `conv_naive` 对齐，`bench_final.py` 则在带填充、空洞、跨步的情形下把 `conv2d` 与 `baseline_naive` 对比。

```bash
python3 bench_final.py
OPENBLAS_NUM_THREADS=6 python3 conv_bench.py
```

想要可比的数字，请把线程数固定为物理核心数（脚本默认 6），并在任何性能结论中注明 CPU、NumPy 版本与 BLAS。

## 参与贡献

欢迎小而专注的补丁，见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。门槛只有两条：参考行为保持简单，以及每个性能结论都附带可复现的测量。

## 路线图

**已完成**

- 支持完整与分块内核的因果多头注意力。
- 具备 1×1、im2col、einsum 与 FFT 路径，并支持 pad/stride/dilation 的二维卷积。
- 两个内置正确性校验的基准脚本。
- 带版本号的文档：[`CHANGELOG.md`](CHANGELOG.md)、[`VERSION`](VERSION)、[`docs/project-status.md`](docs/project-status.md)。

**计划中**

- 无排期。这是一个已关闭的里程碑（2026-09-29）：教学实现已经完成，除非项目的输入或目标发生变化，否则不再继续开发。

如果重新开启，显然的方向正是代码本身已经指向的那几个——反向传播、按 `head_dim` 分批的变体，以及把基准脚本里临时写的校验整理成真正的测试文件。这些目前都不存在。

## 许可证

AGPL-3.0-only，见 [`LICENSE`](LICENSE)。版权所有 (C) 2026 ZisIsNotZis。
