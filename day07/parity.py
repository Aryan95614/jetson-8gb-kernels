#!/usr/bin/env python3
"""
parity.py - Day 7 fused preprocess kernel parity check.

Builds the same synthetic 1280x720x3 uint8 RGB input as preprocess.cu
(same pixel formula, see README.md "Synthetic input pattern" and
fill_synthetic_input() in preprocess.cu), computes a reference
resize + normalize + CHW-transpose in Python, and compares it against
out.bin, the dump written by:

    ./preprocess --dump out.bin

out.bin is the FUSED kernel's output (fused_resize_normalize_chw), which
bilinear-samples straight from the uint8 source and normalizes in one
pass, in float32, with no intermediate uint8-rounded resize buffer.
The reference below is built the same way (float32 all the way through
the resize step) so it matches path B's math, not path A's. Path A
(the three separate kernels) does write an intermediate uint8-rounded
resize buffer, so its output differs from the fused output by up to
about half a uint8 level per pixel; that is an expected, understood
difference between the two paths and is not what this script checks.

Reference resize backend:
  - If OpenCV (cv2) is importable, use cv2.resize(..., interpolation=
    cv2.INTER_LINEAR) on a float32 image. Feeding cv2.resize a float32
    array (rather than uint8) keeps it on OpenCV's floating-point
    bilinear path instead of its 8-bit fixed-point path, so it matches
    the CUDA kernel's pure float32 arithmetic rather than a quantized
    approximation of it.
  - Otherwise (or with --no-cv2), use a pure-numpy bilinear
    implementation with OpenCV's half-pixel-center, clamped-edge
    convention: src = (dst + 0.5) * scale - 0.5, clamped to [0, N-1].
    This is the same convention documented in README.md and implemented
    in the CUDA kernels' bilinear_sample() device function.

Exit status: 0 if max abs diff <= tolerance (default 1e-2), else 1.
"""

import sys
import argparse

import numpy as np

SRC_W, SRC_H = 1280, 720
DST_W, DST_H = 640, 480
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_synthetic_input():
    """Must match fill_synthetic_input() in preprocess.cu exactly:
    value(y, x, c) = (x*7 + y*13 + c*29) mod 256
    Returns HWC uint8, shape (SRC_H, SRC_W, 3).
    """
    xs = np.arange(SRC_W, dtype=np.int32)
    ys = np.arange(SRC_H, dtype=np.int32)
    cs = np.arange(3, dtype=np.int32)
    pattern = (xs[None, :, None] * 7 + ys[:, None, None] * 13 + cs[None, None, :] * 29) % 256
    return pattern.astype(np.uint8)


def bilinear_resize_numpy(img_u8):
    """Pure-numpy bilinear resize matching OpenCV's half-pixel-center,
    clamped-edge convention (see module docstring). img_u8 is HWC
    uint8, SRC_H x SRC_W x 3. Returns float32 HWC, DST_H x DST_W x 3,
    NOT rounded to uint8 - see module docstring for why.
    """
    scale_x = SRC_W / DST_W
    scale_y = SRC_H / DST_H

    dst_x = np.arange(DST_W, dtype=np.float64)
    dst_y = np.arange(DST_H, dtype=np.float64)
    src_x = np.clip((dst_x + 0.5) * scale_x - 0.5, 0, SRC_W - 1)
    src_y = np.clip((dst_y + 0.5) * scale_y - 0.5, 0, SRC_H - 1)

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, SRC_W - 1)
    y1 = np.minimum(y0 + 1, SRC_H - 1)
    fx = (src_x - x0).astype(np.float32)[None, :, None]
    fy = (src_y - y0).astype(np.float32)[:, None, None]

    img_f = img_u8.astype(np.float32)

    top_left = img_f[np.ix_(y0, x0)]
    top_right = img_f[np.ix_(y0, x1)]
    bot_left = img_f[np.ix_(y1, x0)]
    bot_right = img_f[np.ix_(y1, x1)]

    top = top_left + (top_right - top_left) * fx
    bot = bot_left + (bot_right - bot_left) * fx
    out = top + (bot - top) * fy
    return out.astype(np.float32)


def bilinear_resize_cv2(img_u8):
    import cv2
    img_f32 = img_u8.astype(np.float32)
    return cv2.resize(img_f32, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)


def normalize_and_transpose(img_f32_hwc):
    img_f = img_f32_hwc.astype(np.float32) / 255.0
    img_f = (img_f - MEAN[None, None, :]) / STD[None, None, :]
    chw = np.transpose(img_f, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(chw, dtype=np.float32)


def load_dump(path):
    data = np.fromfile(path, dtype=np.float32)
    expected = 3 * DST_H * DST_W
    if data.size != expected:
        print("ERROR: %s has %d float32 values, expected %d (3*%d*%d)" %
              (path, data.size, expected, DST_H, DST_W))
        sys.exit(1)
    return data.reshape(3, DST_H, DST_W)


def main():
    parser = argparse.ArgumentParser(description="Day 7 preprocess kernel parity check")
    parser.add_argument("--dump", default="out.bin",
                         help="path to the fused CUDA output dump (float32 CHW)")
    parser.add_argument("--no-cv2", action="store_true",
                         help="force the pure-numpy bilinear reference even if cv2 is importable")
    parser.add_argument("--tol", type=float, default=1e-2,
                         help="max abs diff tolerance (default 1e-2)")
    args = parser.parse_args()

    img = build_synthetic_input()

    used_cv2 = False
    if not args.no_cv2:
        try:
            resized_f32 = bilinear_resize_cv2(img)
            used_cv2 = True
        except ImportError:
            resized_f32 = bilinear_resize_numpy(img)
    else:
        resized_f32 = bilinear_resize_numpy(img)

    ref_chw = normalize_and_transpose(resized_f32)
    got_chw = load_dump(args.dump)

    diff = np.abs(ref_chw - got_chw)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    backend = "cv2.INTER_LINEAR (float32 input)" if used_cv2 else "pure-numpy bilinear"
    print("parity.py: reference resize backend = %s" % backend)
    print("parity.py: max abs diff  = %.6f" % max_abs_diff)
    print("parity.py: mean abs diff = %.6f" % mean_abs_diff)
    print("parity.py: tolerance     = %.6f" % args.tol)

    if max_abs_diff > args.tol:
        print("parity.py: FAIL (max abs diff exceeds tolerance)")
        sys.exit(1)

    print("parity.py: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
