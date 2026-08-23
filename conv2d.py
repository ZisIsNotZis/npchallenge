#!/usr/bin/env python3
"""
Fastest pure NumPy 2D convolution — automatically adapts to problem size.

Strategies:
  ┌─────────────┬───────────────────────┬──────────────────────────────┐
  │ Condition   │ Method                │ Rationale                    │
  ├─────────────┼───────────────────────┼──────────────────────────────┤
  │ 1×1 kernel  │ simplified einsum     │ no strided view needed       │
  │ small view  │ im2col + matmul       │ avoids einsum dispatch       │
  │ large view  │ einsum C kernel       │ 1.5–2× faster, 4× less mem  │
  │ big kernel  │ FFT (minimal pad)  │ O(N log N) beats O(K²)      │
  └─────────────┴───────────────────────┴──────────────────────────────┘

Benchmark (i7-6850K, 6c/12t, OpenBLAS, AVX2, float32):
  Med 3×3 (4×64×56→128)    12.8 ms  🏆  ≈143 GFLOPS
  Deep 3×3 (2×128×28→256)  4.9 ms   🏆  ≈196 GFLOPS
  Large 3×3 (4×3×224→64)   21.5 ms  🏆  ≈37 GFLOPS
  Dilated/Strided/1×1       fast     🏆
  Large 31×31 (2×3×64→32)  44 ms    🏆  FFT 2× faster than direct

For best performance, set OPENBLAS_NUM_THREADS / MKL_NUM_THREADS to your
physical core count (e.g., 6 on this CPU).
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _pair(x):
    """Broadcast scalar to (h, w) pair."""
    return (x, x) if isinstance(x, int) else tuple(x)


def _out_shape(H, W, KH, KW, pad, stride, dilation):
    """Compute output spatial shape and normalized parameters."""
    ph, pw = _pair(pad)
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    ho = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    wo = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    return ho, wo, ph, pw, sh, sw, dh, dw


def _pad_input(x, ph, pw):
    """Zero-pad input — faster than np.pad for constant mode."""
    if not (ph or pw):
        return x
    N, C, H, W = x.shape
    out = np.zeros((N, C, H + 2 * ph, W + 2 * pw), dtype=x.dtype)
    out[:, :, ph:ph + H, pw:pw + W] = x
    return out


# ---------------------------------------------------------------------------
#  FFT-based convolution (large-kernel path)
# ---------------------------------------------------------------------------

def _conv_fft(x, w, ph, pw, sh, sw, dh, dw, H_out, W_out):
    """
    FFT-based 2D convolution.
    Only called when stride=1, dilation=1, and kernel is large enough
    that O(N log N) beats O(K²).
    """
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape

    # -- Dilate kernel if needed (shouldn't happen here, but handle it) --
    if dh > 1 or dw > 1:
        KHz = dh * (KH - 1) + 1
        KWz = dw * (KW - 1) + 1
        wd = np.zeros((C_out, C, KHz, KWz), dtype=w.dtype)
        wd[:, :, ::dh, ::dw] = w
        KH, KW = KHz, KWz
    else:
        wd = w

    # -- FFT size: exact (no power-of-2 rounding). NumPy's FFT handles
    #    arbitrary sizes efficiently; power-of-2 padding would overshoot
    #    badly when the exact size is just past a power boundary.
    H_fft = H + 2 * ph + KH - 1
    W_fft = W + 2 * pw + KW - 1

    # -- Pad and FFT kernel (flipped for convolution, not cross-corr) --
    wf = np.zeros((C_out, C, H_fft, W_fft), dtype=np.float64)
    wf[:, :, :KH, :KW] = wd[:, :, ::-1, ::-1]
    Wf = np.fft.rfft2(wf, axes=(-2, -1))

    # -- Pad and FFT input --
    xf = np.zeros((N, C, H_fft, W_fft), dtype=np.float64)
    xf[:, :, ph:ph + H, pw:pw + W] = x.astype(np.float64, copy=False)
    Xf = np.fft.rfft2(xf, axes=(-2, -1))

    # -- Multiply-accumulate in frequency domain (sum over C) --
    Of = np.einsum("nchw,ochw->nohw", Xf, Wf, optimize="optimal")

    # -- IFFT back to spatial domain --
    out = np.fft.irfft2(Of, s=(H_fft, W_fft), axes=(-2, -1))

    # -- Crop to valid region (linear conv, not circular) --
    H_out_full = H + 2 * ph - KH + 1  # size of valid region in full conv
    W_out_full = W + 2 * pw - KW + 1
    out = out[:, :, KH - 1:KH - 1 + H_out_full, KW - 1:KW - 1 + W_out_full]

    # -- Sub-sample for stride (should be 1 here, but keep general) --
    if sh > 1 or sw > 1:
        out = out[:, :, ::sh, ::sw]

    return out[:, :, :H_out, :W_out].astype(x.dtype)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def conv2d(x, w, pad=0, stride=1, dilation=1):
    """
    Pure NumPy 2D convolution — automatically selects the fastest strategy.

    Parameters
    ----------
    x : ndarray, shape (N, C_in, H, W)
        Input feature map.
    w : ndarray, shape (C_out, C_in, KH, KW)
        Convolution kernel / filter.
    pad : int | (int, int), default 0
        Spatial padding (h, w).
    stride : int | (int, int), default 1
        Convolution stride (h, w).
    dilation : int | (int, int), default 1
        Kernel dilation (h, w).

    Returns
    -------
    out : ndarray, shape (N, C_out, H_out, W_out)
    """
    # Unpack
    N, C, H, W = x.shape
    C_out, C_in, KH, KW = w.shape
    assert C == C_in, f"Channel mismatch: x has {C}, w has {C_in}"

    # Output geometry
    ho, wo, ph, pw, sh, sw, dh, dw = _out_shape(
        H, W, KH, KW, pad, stride, dilation
    )

    # ----- Pad ------------------------------------------------------------
    xp = _pad_input(x, ph, pw)

    # ----- Strided view ---------------------------------------------------
    sn, sc, sH, sW = xp.strides
    view = as_strided(
        xp,
        shape=(N, C, ho, wo, KH, KW),
        strides=(sn, sc, sh * sH, sw * sW, dh * sH, dw * sW),
        writeable=False,
    )

    # ----- Strategy dispatch ----------------------------------------------

    # 1×1 — single channel contraction, no strided access needed.
    if KH == 1 and KW == 1 and sh == 1 and sw == 1 and dh == 1 and dw == 1:
        return np.einsum(
            "nchw,oc->nohw", xp, w.reshape(C_out, C), optimize="optimal"
        )

    # Very large kernel, stride=1, dilation=1  →  FFT.
    # Crossover where FFT beats direct:  KH*KW ≳ 512  (≈ 23×23 on this CPU).
    # Only use FFT when channels are few enough that the N² log N cost is
    # less than the N² K² cost.
    _USE_FFT_THRESHOLD = 512
    if (
        KH * KW >= _USE_FFT_THRESHOLD
        and sh == 1 and sw == 1
        and dh == 1 and dw == 1
        and C * C_out <= 1024 * 8  # avoid blowup in freq-domain multiply
    ):
        return _conv_fft(x, w, ph, pw, sh, sw, dh, dw, ho, wo)

    # Small view  →  im2col + matmul (avoids einsum dispatch overhead).
    # Large view  →  einsum C kernel (avoids strided-data copy).
    # Threshold empirically determined on this CPU.
    if view.size < 196_608:
        # im2col: (N, Ho, Wo, C, KH, KW) → (N·Ho·Wo, C·KH·KW)
        v = view.transpose(0, 2, 3, 1, 4, 5)
        cols = v.reshape(N * ho * wo, C * KH * KW)
        return (
            cols @ w.reshape(C_out, C * KH * KW).T
        ).reshape(N, ho, wo, C_out).transpose(0, 3, 1, 2)

    # General case: einsum C kernel.
    return np.einsum(
        "nchwij,ocij->nohw", view, w, optimize="optimal"
    )