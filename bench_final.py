#!/usr/bin/env python3
"""
Comprehensive benchmark for conv2d against all baseline implementations.
"""
import os; os.environ['OPENBLAS_NUM_THREADS'] = '6'
import ctypes
try:
    lib = ctypes.CDLL('libopenblas.so.0'); lib.openblas_set_num_threads(6)
except Exception:
    pass

import numpy as np
from numpy.lib.stride_tricks import as_strided
from time import perf_counter
import gc, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from conv2d import conv2d

# ─── Baseline implementations ────────────────────────────────────────────────

def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def _out_shape(H, W, KH, KW, pad, stride, dilation):
    ph, pw = _pair(pad); sh, sw = _pair(stride); dh, dw = _pair(dilation)
    return (H+2*ph-dh*(KH-1)-1)//sh+1, (W+2*pw-dw*(KW-1)-1)//sw+1, ph, pw, sh, sw, dh, dw

def _pad(x, ph, pw):
    if not (ph or pw): return x
    out = np.zeros((x.shape[0],x.shape[1],x.shape[2]+2*ph,x.shape[3]+2*pw),dtype=x.dtype)
    out[:,:,ph:ph+x.shape[2],pw:pw+x.shape[3]] = x
    return out

def baseline_naive(x, w, pad=0, stride=1, dilation=1):
    N, C, H, W = x.shape; C_out, C_in, KH, KW = w.shape
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

def baseline_im2col(x, w, pad=0, stride=1, dilation=1):
    N, C, H, W = x.shape; C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s), writeable=False)
    v = v.transpose(0, 2, 3, 1, 4, 5)
    cols = v.reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (cols @ wf.T).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

def baseline_im2col_v2(x, w, pad=0, stride=1, dilation=1):
    N, C, H, W = x.shape; C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, ho, wo, C, KH, KW),
                   strides=(sn, sh*sh_s, sw*sw_s, sc, dh*sh_s, dw*sw_s), writeable=False)
    cols = v.reshape(N*ho*wo, C*KH*KW)
    wf = w.reshape(C_out, C*KH*KW)
    return (cols @ wf.T).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

def baseline_einsum(x, w, pad=0, stride=1, dilation=1):
    N, C, H, W = x.shape; C_out, C_in, KH, KW = w.shape
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(H, W, KH, KW, pad, stride, dilation)
    xp = _pad(x, ph, pw)
    sn, sc, sh_s, sw_s = xp.strides
    v = as_strided(xp, shape=(N, C, ho, wo, KH, KW),
                   strides=(sn, sc, sh*sh_s, sw*sw_s, dh*sh_s, dw*sw_s), writeable=False)
    return np.einsum('nchwij,ocij->nohw', v, w, optimize='optimal')

# ─── Benchmark ────────────────────────────────────────────────────────────────

def run():
    implementations = [
        ('conv2d (ours)',   conv2d),
        ('einsum',          baseline_einsum),
        ('im2col',          baseline_im2col),
        ('im2col_v2',       baseline_im2col_v2),
    ]

    shapes = [
        ("Small 3×3, s1",   (1, 3, 32, 32),   (16, 3, 3, 3),   1, 1, 1),
        ("Small 5×5, s1",   (1, 3, 32, 32),   (16, 3, 5, 5),   2, 1, 1),
        ("Med 3×3, s1",     (4, 64, 56, 56),  (128, 64, 3, 3), 1, 1, 1),
        ("Med 7×7, s2",     (4, 3, 64, 64),   (64, 3, 7, 7),   3, 2, 1),
        ("Large 3×3, s1",   (4, 3, 224, 224), (64, 3, 3, 3),   1, 1, 1),
        ("Dilated 3×3, d2", (4, 3, 64, 64),   (16, 3, 3, 3),   2, 1, 2),
        ("Strided 5×5, s2", (4, 3, 128, 128), (32, 3, 5, 5),   2, 2, 1),
        ("1×1 conv",        (4, 32, 64, 64),  (64, 32, 1, 1),  0, 1, 1),
        ("Deep 3×3, s1",    (2, 128, 28, 28), (256, 128, 3, 3),1, 1, 1),
        ("Large 7×7, s1",   (2, 3, 128, 128), (32, 3, 7, 7),   3, 1, 1),
    ]

    # --- Correctness ---
    print("✓ Correctness verification (small case, pad=1, s=1, d=1)...")
    x_v = np.random.randn(1, 2, 8, 8).astype(np.float32)
    w_v = np.random.randn(4, 2, 3, 3).astype(np.float32)
    ref = baseline_naive(x_v, w_v, 1, 1, 1)
    # Also test with dilation and stride
    ref_d = baseline_naive(x_v, w_v, 2, 2, 2)
    ref_s = baseline_naive(x_v, w_v, 1, 2, 1)

    for name, fn in implementations:
        ok = np.max(np.abs(fn(x_v, w_v, 1, 1, 1) - ref)) < 1e-4
        ok_d = np.max(np.abs(fn(x_v, w_v, 2, 2, 2) - ref_d)) < 1e-4
        ok_s = np.max(np.abs(fn(x_v, w_v, 1, 2, 1) - ref_s)) < 1e-4
        status = "✓" if (ok and ok_d and ok_s) else "✗"
        print(f"  {name:20s}: {status}")
        if not ok: print(f"    FAIL base, diff={np.max(np.abs(fn(x_v,w_v,1,1,1)-ref)):.2e}")
        if not ok_d: print(f"    FAIL dilation, diff={np.max(np.abs(fn(x_v,w_v,2,2,2)-ref_d)):.2e}")
        if not ok_s: print(f"    FAIL stride, diff={np.max(np.abs(fn(x_v,w_v,1,2,1)-ref_s)):.2e}")

    # --- Timing ---
    print(f"\n{'='*96}")
    print(f"  {'Shape':<28s}  {'conv2d (ours)':>14s}  {'einsum':>12s}  {'im2col':>12s}  {'im2col_v2':>12s}  {'winner':>10s}")
    print(f"{'='*96}")

    wins = {}
    for name, x_shape, w_shape, pad, stride, dilation in shapes:
        x = np.random.randn(*x_shape).astype(np.float32)
        w = np.random.randn(*w_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            for _, fn in implementations:
                fn(x, w, pad, stride, dilation)

        results = []
        for impl_name, fn in implementations:
            times = []
            for _ in range(20):
                t0 = perf_counter()
                fn(x, w, pad, stride, dilation)
                times.append(perf_counter() - t0)
            t_med = np.median(times) * 1000
            results.append((impl_name, t_med))

        best = min(r[1] for r in results)
        winner = [r[0] for r in results if r[1] == best][0]
        wins[winner] = wins.get(winner, 0) + 1

        suffix = ""
        for rname, rt in results:
            if rname == 'conv2d (ours)':
                ratio = rt / best
                suffix = f"  x{ratio:.2f} vs best" if ratio > 1.01 else "  🏆"

        print(f"  {name:<28s}  {results[0][1]:>8.2f} ms{'':>2s}  "
              f"{results[1][1]:>8.2f} ms{'':>2s}  "
              f"{results[2][1]:>8.2f} ms{'':>2s}  "
              f"{results[3][1]:>8.2f} ms{'':>2s}  {suffix}")

    print(f"{'='*96}")
    print(f"  Wins: {wins}")
    print(f"{'='*96}")

    # Additional info
    print(f"\n  GFLOPS for selected cases (using conv2d):")
    for name, x_shape, w_shape, pad, stride, dilation in shapes:
        x = np.random.randn(*x_shape).astype(np.float32)
        w = np.random.randn(*w_shape).astype(np.float32)
        N, C, H, W = x_shape
        C_out, C_in, KH, KW = w_shape
        ho, wo = _out_shape(H, W, KH, KW, pad, stride, dilation)[:2]
        macs = 2 * N * C_out * C * KH * KW * ho * wo
        times = []
        for _ in range(10):
            t0 = perf_counter()
            conv2d(x, w, pad, stride, dilation)
            times.append(perf_counter() - t0)
        t_med = np.median(times)
        gflops = macs / t_med / 1e9
        print(f"  {name:<28s}  {macs/1e9:>6.1f} GFLOP  {t_med*1000:>8.2f} ms  {gflops:>5.1f} GFLOPS")

if __name__ == '__main__':
    run()