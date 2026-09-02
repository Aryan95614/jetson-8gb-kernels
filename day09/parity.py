#!/usr/bin/env python3
"""
parity.py - Day 9 parity check.

The rolling synthetic frame pattern used by host_py.py / host_rs is:
  value(frame, y, x, c) = (x*7 + y*13 + c*29 + frame) mod 256
At frame index 0 this is numerically IDENTICAL to Day 7's static
pattern (x*7 + y*13 + c*29) mod 256. So a single call into libfused.so
with frame 0 can be checked against Day 7's reference implementation
directly, with no reimplementation of the bilinear/normalize math
needed on the Python side other than as a fallback (see below).

This script:
  1. Loads libfused.so via ctypes, runs fused_init/upload/run/download
     once on frame 0.
  2. Builds a reference frame-0 output using Day 7's own reference
     functions if ../day07/parity.py is importable (build_synthetic_
     input, bilinear_resize_numpy, normalize_and_transpose); otherwise
     falls back to an inline reimplementation of the same formula
     (half-pixel-center clamped bilinear, then normalize, then CHW).
  3. Prints max abs diff / mean abs diff and PASS/FAIL against the
     same 1e-2 tolerance Day 7 uses.

Usage: python3 parity.py
"""

import ctypes
import os
import sys

import numpy as np

SRC_W, SRC_H = 1280, 720
DST_W, DST_H = 640, 480
CH = 3
TOL = 1e-2

HERE = os.path.dirname(os.path.abspath(__file__))
DAY07_DIR = os.path.normpath(os.path.join(HERE, "..", "day07"))


# ---------------------------------------------------------------------
# Fallback reference (used only if ../day07/parity.py is not present).
# Same formula as Day 7's parity.py: half-pixel-center clamped bilinear,
# ImageNet mean/std normalize, HWC -> CHW.
# ---------------------------------------------------------------------
def build_synthetic_input_fallback():
    xs = np.arange(SRC_W, dtype=np.int32)
    ys = np.arange(SRC_H, dtype=np.int32)
    cs = np.arange(CH, dtype=np.int32)
    pattern = (xs[None, :, None] * 7 + ys[:, None, None] * 13 + cs[None, None, :] * 29) % 256
    return pattern.astype(np.uint8)


def bilinear_resize_numpy_fallback(img_u8):
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


def normalize_and_transpose_fallback(img_f32_hwc):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = img_f32_hwc.astype(np.float32) / 255.0
    img_f = (img_f - mean[None, None, :]) / std[None, None, :]
    chw = np.transpose(img_f, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32)


def load_reference():
    """Prefer Day 7's own reference functions (exact same file the CUDA
    kernel was validated against on Day 7); fall back to the inline
    reimplementation above if that file is not present next to this
    package (e.g. this directory copied somewhere standalone).
    """
    day07_parity = os.path.join(DAY07_DIR, "parity.py")
    if os.path.isfile(day07_parity):
        sys.path.insert(0, DAY07_DIR)
        import parity as day07_ref  # type: ignore  # ../day07/parity.py

        return (
            day07_ref.build_synthetic_input,
            day07_ref.bilinear_resize_numpy,
            day07_ref.normalize_and_transpose,
            "../day07/parity.py",
        )
    return (
        build_synthetic_input_fallback,
        bilinear_resize_numpy_fallback,
        normalize_and_transpose_fallback,
        "inline fallback (../day07/parity.py not found)",
    )


def run_shim_frame0():
    lib_path = os.environ.get("FUSED_LIB", os.path.join(HERE, "libfused.so"))
    lib = ctypes.CDLL(lib_path)
    lib.fused_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.fused_init.restype = ctypes.c_int
    lib.fused_upload.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.fused_upload.restype = ctypes.c_int
    lib.fused_run.argtypes = []
    lib.fused_run.restype = ctypes.c_int
    lib.fused_download.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.fused_download.restype = ctypes.c_int
    lib.fused_free.argtypes = []
    lib.fused_free.restype = None

    if lib.fused_init(SRC_W, SRC_H, DST_W, DST_H) != 0:
        print("parity.py: fused_init failed", file=sys.stderr)
        sys.exit(1)

    xs = np.arange(SRC_W, dtype=np.int32)
    ys = np.arange(SRC_H, dtype=np.int32)
    cs = np.arange(CH, dtype=np.int32)
    frame0 = ((xs[None, :, None] * 7 + ys[:, None, None] * 13 + cs[None, None, :] * 29) % 256)
    frame0 = frame0.astype(np.uint8)
    ptr = frame0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    if lib.fused_upload(ptr) != 0:
        print("parity.py: fused_upload failed", file=sys.stderr)
        sys.exit(1)
    if lib.fused_run() != 0:
        print("parity.py: fused_run failed", file=sys.stderr)
        sys.exit(1)

    out = np.zeros((CH, DST_H, DST_W), dtype=np.float32)
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if lib.fused_download(out_ptr) != 0:
        print("parity.py: fused_download failed", file=sys.stderr)
        sys.exit(1)

    lib.fused_free()
    return out


def main():
    build_synth, resize, normalize, src_desc = load_reference()
    print("parity.py: reference source = %s" % src_desc)

    img = build_synth()
    resized = resize(img)
    ref_chw = normalize(resized)

    got_chw = run_shim_frame0()

    diff = np.abs(ref_chw - got_chw)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    print("parity.py: max abs diff  = %.6f" % max_abs_diff)
    print("parity.py: mean abs diff = %.6f" % mean_abs_diff)
    print("parity.py: tolerance     = %.6f" % TOL)

    if max_abs_diff > TOL:
        print("parity.py: FAIL (max abs diff exceeds tolerance)")
        sys.exit(1)

    print("parity.py: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
