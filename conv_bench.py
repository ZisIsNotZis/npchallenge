#!/usr/bin/env python3
"""
Fast pure NumPy 2D convolution — benchmark + optimization loop.
Intel i7-6850K (Haswell, 6c/12t), OpenBLAS, AVX2.

Usage:  OPENBLAS_NUM_THREADS=6 python3 conv_bench.py
"""
import os; os.environ['OPENBLAS_NUM_THREADS'] = '6'
import ctypes
try:
    lib = ctypes.CDLL('libopenblas.so.0')
    lib.openblas_set_num_threads(6)
except Exception:
    pass

import numpy as np
from numpy.lib.stride_tricks import as_strided
from time import perf_counter
import gc, warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, linewidth=120)

# ─── Utilities ────────────────────────────────────────────────────────────────

def _to_tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def _out_shape(H, W, KH, KW, pad, stride, dilation):
    ph, pw = _to_tuple(pad)
    sh, sw = _to_tuple(stride)
    dh, dw = _to_tuple(dilation)
    ho = (H + 2*ph - dh*(KH-1) - 1)//sh + 1
    wo = (W + 2*pw - dw*(KW-1) - 1)//sw + 1
    return ho, wo, ph, pw, sh, sw, dh, dw

def _pad(x, ph, pw):
    """Fast zero-padding."""
    if not (ph or pw):
        return x
    N, C, H, W = x.shape
    out = np.zeros((N, C, H + 2*ph, W + 2*pw), dtype=x.dtype)
    out[:, :, ph:ph+H, pw:pw+W] = x
    return out

# ─── Implementations ─────────────────────────────────────────────────────────

# --- Baseline: naive 6-loop (only for tiny verification) ---
def conv_naive(x, w, pad=0, stride=1, dilation=1):
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    out = np.zeros((N, C_out, ho, wo), dtype=x.dtype)
    for n in range(N):
        for co in range(C_out):
            for ci in range(C):
                for oh in range(ho):
                    ih0 = oh * sh
                    for ow in range(wo):
                        iw0 = ow * sw
                        s = 0.0
                        for kh in range(KH):
                            ih = ih0 + kh * dh
                            for kw in range(KW):
                                iw = iw0 + kw * dw
                                s += xp[n, ci, ih, iw] * w[co, ci, kh, kw]
                        out[n, co, oh, ow] += s
    return out

# --- Core im2col variants ---

def conv_im2col(x, w, pad=0, stride=1, dilation=1):
    """Standard im2col + matmul."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    # View as (N, C, ho, wo, KH, KW), then transpose to (N, ho, wo, C, KH, KW)
    # so that reshape to (N*ho*wo, C*KH*KW) gives the correct im2col layout.
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    cols = v.transpose(0, 2, 3, 1, 4, 5).reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (cols @ wf.T).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

def conv_im2col_v2(x, w, pad=0, stride=1, dilation=1):
    """im2col with (N, H_out, W_out, C, KH, KW) layout."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, ho, wo, C, KH, KW),
                   strides=(sn, sh*sh_s, sw*sw_s, sc, dh*sh_s, dw*sw_s),
                   writeable=False)
    cols = v.reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (cols @ wf.T).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

def conv_im2col_out(x, w, pad=0, stride=1, dilation=1):
    """im2col + matmul with pre-allocated output buffer."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    cols = v.transpose(0, 2, 3, 1, 4, 5).reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    out = np.empty((N*ho*wo, C_out), dtype=x.dtype)
    np.matmul(cols, wf.T, out=out)
    return out.reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

def conv_im2col_not(x, w, pad=0, stride=1, dilation=1):
    """im2col: wf @ cols.T avoids wf.T."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    cols = v.transpose(0, 2, 3, 1, 4, 5).reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (wf @ cols.T).reshape(C_out, N, ho, wo).transpose(1, 0, 2, 3)

def conv_tensordot(x, w, pad=0, stride=1, dilation=1):
    """tensordot on strided view."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    return np.tensordot(v, w, axes=([1,4,5],[1,2,3])).transpose(0, 3, 1, 2)

def conv_einsum_opt(x, w, pad=0, stride=1, dilation=1):
    """einsum with optimize='optimal'."""
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    return np.einsum('nchwij,ocij->nohw', v, w, optimize='optimal')

# ─── Optimized: fused pad + as_strided in one go ──────────────────────────────

def conv_fused(x, w, pad=0, stride=1, dilation=1):
    """
    Fused: create padded array and as_strided view in one step.
    Instead of np.pad/np.zeros + as_strided, we use as_strided directly
    on the unpadded input with adjusted offsets.
    WARNING: as_strided with negative strides can segfault if misused.
    Only works when padding fits within the same memory page.
    """
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    ph, pw = _to_tuple(pad)
    sh, sw = _to_tuple(stride)
    dh, dw = _to_tuple(dilation)
    ho = (H + 2*ph - dh*(KH-1) - 1)//sh + 1
    wo = (W + 2*pw - dw*(KW-1) - 1)//sw + 1

    if ph or pw:
        # Explicit pad — as_strided tricks with negative strides are unsafe
        xp = np.zeros((N, C, H + 2*ph, W + 2*pw), dtype=x.dtype)
        xp[:, :, ph:ph+H, pw:pw+W] = x
    else:
        xp = x

    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s),
                   writeable=False)
    cols = v.transpose(0, 2, 3, 1, 4, 5).reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (cols @ wf.T).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(impls, shapes, num_warmup=5, num_trials=15):
    """Run benchmark: verify correctness, then time each impl."""

    # Verify correctness on a tiny case first
    print("✓ Correctness verification (small case)...")
    x_v = np.random.randn(1, 2, 8, 8).astype(np.float32)
    w_v = np.random.randn(4, 2, 3, 3).astype(np.float32)
    ref = conv_naive(x_v, w_v, 1, 1, 1)
    for name, fn in impls:
        if name == 'naive':
            continue
        out = fn(x_v, w_v, 1, 1, 1)
        diff = np.max(np.abs(out - ref))
        ok = diff < 1e-4
        print(f"  {name:22s}: {'✓' if ok else '✗ FAIL'}  (max diff={diff:.2e})")
        if not ok:
            print(f"    WARNING: {name} failed correctness test!")

    # Also verify with dilation
    x_v2 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    w_v2 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    ref2 = conv_naive(x_v2, w_v2, 2, 2, 2)
    for name, fn in impls:
        if name == 'naive':
            continue
        out = fn(x_v2, w_v2, 2, 2, 2)
        diff = np.max(np.abs(out - ref2))
        ok = diff < 1e-4
        if not ok:
            print(f"  {name:22s}: ✗ DILATION FAIL (max diff={diff:.2e})")

    # Verify with stride != 1
    x_v3 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    w_v3 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    ref3 = conv_naive(x_v3, w_v3, 1, 2, 1)
    for name, fn in impls:
        if name == 'naive':
            continue
        out = fn(x_v3, w_v3, 1, 2, 1)
        diff = np.max(np.abs(out - ref3))
        ok = diff < 1e-4
        if not ok:
            print(f"  {name:22s}: ✗ STRIDE FAIL (max diff={diff:.2e})")

    # Benchmark each shape
    fastest_counts = {}
    for name, x_shape, w_shape, pad, stride, dilation in shapes:
        ho, wo = _out_shape(x_shape[2], x_shape[3], w_shape[2], w_shape[3], pad, stride, dilation)[:2]
        print(f"\n{'─'*70}")
        print(f"  {name}")
        print(f"  x={x_shape}  w={w_shape}  pad={pad}  stride={stride}  dilation={dilation}")
        print(f"  → ({x_shape[0]}, {w_shape[0]}, {ho}, {wo})  "
              f"MACs={x_shape[0]*w_shape[0]*x_shape[1]*w_shape[2]*w_shape[3]*ho*wo/1e6:.0f}M")
        print(f"{'─'*70}")

        x = np.random.randn(*x_shape).astype(np.float32)
        w = np.random.randn(*w_shape).astype(np.float32)

        results = []
        for impl_name, impl_fn in impls:
            if impl_name == 'naive':
                continue
            gc.collect()
            # Warmup
            for _ in range(num_warmup):
                _ = impl_fn(x, w, pad, stride, dilation)
            # Time
            times = []
            for _ in range(num_trials):
                gc.collect()
                t0 = perf_counter()
                _ = impl_fn(x, w, pad, stride, dilation)
                times.append(perf_counter() - t0)
            t_min = np.min(times) * 1000
            t_mean = np.mean(times) * 1000
            t_std = np.std(times) * 1000
            results.append((impl_name, t_min, t_mean, t_std))

        # Sort by mean time
        results.sort(key=lambda r: r[2])
        best_time = results[0][2]
        for impl_name, t_min, t_mean, t_std in results:
            ratio = t_mean / best_time
            mark = ' 🏆' if ratio <= 1.01 else ''
            print(f"  {impl_name:22s}: {t_mean:8.2f} ± {t_std:5.2f} ms  "
                  f"(min: {t_min:.2f}, x{ratio:.2f}){mark}")
            if ratio <= 1.01:
                fastest_counts[impl_name] = fastest_counts.get(impl_name, 0) + 1

    # Summary
    print(f"\n{'='*70}")
    print("  Fastest implementation counts:")
    for name, count in sorted(fastest_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}x fastest")
    print(f"{'='*70}")

    return fastest_counts


if __name__ == '__main__':
    impls = [
        ('naive',           conv_naive),
        ('im2col',          conv_im2col),
        ('im2col_v2',       conv_im2col_v2),
        ('im2col_out',      conv_im2col_out),
        ('im2col_not',      conv_im2col_not),
        ('tensordot',       conv_tensordot),
        ('einsum_opt',      conv_einsum_opt),
        ('fused',           conv_fused),
    ]

    shapes = [
        ("Small 3x3",       (1, 3, 32, 32),   (16, 3, 3, 3),   1, 1, 1),
        ("Small 5x5",       (1, 3, 32, 32),   (16, 3, 5, 5),   2, 1, 1),
        ("Med 3x3",         (4, 64, 56, 56),  (128, 64, 3, 3), 1, 1, 1),
        ("Med 7x7",         (4, 3, 64, 64),   (64, 3, 7, 7),   3, 2, 1),
        ("Large 3x3",       (4, 3, 224, 224), (64, 3, 3, 3),   1, 1, 1),
        ("Dilated 3x3",     (4, 3, 64, 64),   (16, 3, 3, 3),   2, 1, 2),
        ("Strided 5x5",     (4, 3, 128, 128), (32, 3, 5, 5),   2, 2, 1),
        ("1x1 Conv",        (4, 32, 64, 64),  (64, 32, 1, 1),  0, 1, 1),
        ("Deep 3x3",        (2, 128, 28, 28), (256, 128, 3, 3),1, 1, 1),
        ("Large 7x7",       (2, 3, 128, 128), (32, 3, 7, 7),   3, 1, 1),
    ]

    benchmark(impls, shapes, num_warmup=5, num_trials=15)