#!/usr/bin/env python3
"""
host_py.py - Day 9: Python host launching the Day 7 fused CUDA kernel
(now built as libfused.so, see shim.cu) via ctypes.

Two experiments, same kernel, same frame pattern as host_rs (the Rust
host, see host_rs/src/main.rs):

  (i) tight loop: 1000 iterations of upload+run, per-iteration wall
      time via time.perf_counter(). Reports mean/p50/p99/max in
      microseconds after discarding the first 50 iterations (warmup).

  (ii) paced 30 Hz loop: 600 ticks, each tick targets start + i/30 s,
       sleeps close to the deadline then spins the last 0.5 ms, does
       upload+run, and records how late the tick actually started
       relative to its target (the "jitter"). Reports jitter
       mean/p50/p99/max in microseconds and a count of ticks that
       started more than 2 ms late.

Also records startup_ms: wall time from as close to process start as a
pure Python script can measure (time.perf_counter() at the very top of
this file, before any other import) to the first completed fused_run()
call.

Prints one line prefixed "RESULT_JSON: " with everything above, so
run.sh can extract it without scraping human-readable text. See
host_rs/src/main.rs for the Rust side, which prints the identical
schema.
"""

import time

_T0 = time.perf_counter()  # as close to "process start" as pure Python gets

import ctypes
import json
import os
import sys

import numpy as np

SRC_W, SRC_H = 1280, 720
DST_W, DST_H = 640, 480
CH = 3

TIGHT_ITERS = 1000
TIGHT_WARMUP = 50

PACED_TICKS = 600
PACED_HZ = 30.0
LATE_THRESHOLD_US = 2000.0  # 2 ms
SPIN_TAIL_S = 0.0005        # sleep to within 0.5 ms of the deadline, then spin

LIB_PATH = os.environ.get("FUSED_LIB", "./libfused.so")


def load_lib():
    lib = ctypes.CDLL(LIB_PATH)
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
    return lib


def build_base_pattern():
    """Frame-0 term of the rolling synthetic pattern:
      value(frame, y, x, c) = (x*7 + y*13 + c*29 + frame) mod 256
    Precomputed once as int32 (pre-mod, pre-rolling-index); each frame
    adds its index and re-mods. Shape (SRC_H, SRC_W, CH), matches the
    HWC layout the kernel expects.
    """
    xs = np.arange(SRC_W, dtype=np.int32)
    ys = np.arange(SRC_H, dtype=np.int32)
    cs = np.arange(CH, dtype=np.int32)
    base = (xs[None, :, None] * 7 + ys[:, None, None] * 13 + cs[None, None, :] * 29) % 256
    return base.astype(np.int32)


def make_frame(base, frame_idx):
    """Rolling frame, NOT timed (built before the timed upload+run region)."""
    return ((base + frame_idx) % 256).astype(np.uint8)


def percentile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = int(p * (n - 1))
    return sorted_vals[idx]


def stats_us(samples_us):
    s = sorted(samples_us)
    if not s:
        return {"mean_us": 0.0, "p50_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    return {
        "mean_us": sum(s) / len(s),
        "p50_us": percentile(s, 0.50),
        "p99_us": percentile(s, 0.99),
        "max_us": s[-1],
    }


def main():
    lib = load_lib()
    rc = lib.fused_init(SRC_W, SRC_H, DST_W, DST_H)
    if rc != 0:
        print("host_py.py: fused_init failed, rc=%d" % rc, file=sys.stderr)
        sys.exit(1)

    base = build_base_pattern()

    # ---- Experiment (i): tight loop ----
    tight_samples_us = []
    startup_ms = None
    for i in range(TIGHT_ITERS):
        frame = make_frame(base, i)
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # Error checks happen AFTER t1, not inline -- keeping the branch out
        # of the timed region matches host_rs, which hoists its up_rc/run_rc
        # checks past t_end for the same reason (see README, "Benchmark
        # methodology": only upload+run itself should be inside the clock).
        t0 = time.perf_counter()
        up_rc = lib.fused_upload(ptr)
        run_rc = lib.fused_run()
        t1 = time.perf_counter()
        if up_rc != 0 or run_rc != 0:
            print("host_py.py: fused_upload/fused_run failed at iter %d "
                  "(up_rc=%d run_rc=%d)" % (i, up_rc, run_rc), file=sys.stderr)
            sys.exit(1)

        if startup_ms is None:
            startup_ms = (t1 - _T0) * 1000.0
        tight_samples_us.append((t1 - t0) * 1e6)

    tight_stats = stats_us(tight_samples_us[TIGHT_WARMUP:])

    # ---- Experiment (ii): paced 30 Hz loop ----
    jitter_samples_us = []
    work_samples_us = []
    late_ticks = 0
    paced_start = time.perf_counter()
    for i in range(PACED_TICKS):
        target = paced_start + i / PACED_HZ

        now = time.perf_counter()
        sleep_for = target - now - SPIN_TAIL_S
        if sleep_for > 0:
            time.sleep(sleep_for)
        while time.perf_counter() < target:
            pass  # spin the last stretch, sleep() granularity is too coarse

        tick_start = time.perf_counter()
        jitter_us = (tick_start - target) * 1e6
        jitter_samples_us.append(jitter_us)
        if jitter_us > LATE_THRESHOLD_US:
            late_ticks += 1

        frame = make_frame(base, TIGHT_ITERS + i)
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        # Same rule as the tight loop: error checks happen after w1.
        w0 = time.perf_counter()
        up_rc = lib.fused_upload(ptr)
        run_rc = lib.fused_run()
        w1 = time.perf_counter()
        if up_rc != 0 or run_rc != 0:
            print("host_py.py: paced-loop upload/run failed at tick %d "
                  "(up_rc=%d run_rc=%d)" % (i, up_rc, run_rc), file=sys.stderr)
            sys.exit(1)
        work_samples_us.append((w1 - w0) * 1e6)

    jitter_stats = stats_us(jitter_samples_us)
    work_mean_us = (sum(work_samples_us) / len(work_samples_us)) if work_samples_us else 0.0

    lib.fused_free()

    result = {
        "host": "python",
        "startup_ms": startup_ms if startup_ms is not None else 0.0,
        "tight_loop": {
            "iters": TIGHT_ITERS,
            "warmup_discarded": TIGHT_WARMUP,
            "mean_us": tight_stats["mean_us"],
            "p50_us": tight_stats["p50_us"],
            "p99_us": tight_stats["p99_us"],
            "max_us": tight_stats["max_us"],
        },
        "paced_loop": {
            "ticks": PACED_TICKS,
            "hz": PACED_HZ,
            "jitter_p50_us": jitter_stats["p50_us"],
            "jitter_p99_us": jitter_stats["p99_us"],
            "jitter_max_us": jitter_stats["max_us"],
            "late_ticks_over_2ms": late_ticks,
            "work_mean_us": work_mean_us,
        },
    }
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
