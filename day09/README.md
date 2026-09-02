# Day 9: Rust as the host, CUDA as the device

Target: Jetson Orin Nano 8GB Super (Ampere, sm_87), CUDA 12.6, JetPack 6.2,
aarch64 Linux, Python 3.10 with numpy 1.26, no sudo. Rust stable via
rustup, installed into `~/.cargo`, no external crates (std only).

## What this shows

Day 7 (`../day07/`) fused three preprocessing kernels into one and
showed that fusion cuts launch/DRAM-round-trip overhead. Day 9 holds
that fused kernel fixed and changes only the host language that
launches it, to answer a different question: does the host language
change how fast the GPU work runs, or only how consistently the CPU
side shows up to launch it?

Same kernel (`fused_resize_normalize_chw`, copied byte-for-byte from
Day 7's `preprocess.cu` into `shim.cu`), same problem size (1280x720x3
uint8 HWC in, 3x480x640 float32 CHW out), same synthetic input
generator, same two experiments, run from two different processes:

- **Python host** (`host_py.py`): ctypes calls into `libfused.so`.
- **Rust host** (`host_rs/`): a `extern "C"` FFI call into the same
  `libfused.so`, no Python runtime involved at all.

Both hosts run:

1. **Tight loop**: 1000 iterations of upload+run, back to back, no
   pacing. Measures per-iteration wall time; the first 50 iterations
   are discarded as warmup before computing mean/p50/p99/max.
2. **Paced 30 Hz loop**: 600 ticks, each targeting `start + i/30 s`.
   Each tick sleeps close to its deadline (`std::thread::sleep` /
   `time.sleep`) and then busy-spins the last 0.5 ms for precision,
   does upload+run, and records how late the tick actually started
   relative to its target -- that lateness is the **jitter** this
   experiment is measuring. A count of ticks that started more than
   2 ms late is also reported.

The GPU work in both experiments is byte-identical: same kernel, same
launch geometry, same stream-synchronize-per-call semantics. Anything
that differs between the two hosts' results is host-side effect, not
device-side effect.

## The claim under test

**Rust does not make the GPU faster; it kills host jitter** (no GIL,
no hidden allocator in the hot path, no interpreter dispatch between
the FFI call and the C function).

This predicts:

- **Tight-loop p50/p99 should be close to identical between the two
  hosts.** The tight loop is GPU-bound (`cudaStreamSynchronize` blocks
  until the kernel finishes); the host's job in that loop is just
  "call two C functions back to back," which is cheap in both
  languages. If Python's p50/p99 come out noticeably higher here, that
  is evidence of per-call ctypes/GIL overhead large enough to show up
  next to kernel time, not evidence against the fusion work from Day 7.
- **Paced-loop jitter should differ more than tight-loop timing does.**
  The paced loop's jitter is a measurement of *scheduling precision*:
  how close the host actually gets to hitting `sleep`+spin deadlines
  under real OS scheduling, with (Python) or without (Rust) a GIL and
  a garbage-collected/refcounted runtime doing background work between
  ticks. This is where a host-language difference, if there is one,
  should show up most clearly.

**If tight-loop p50/p99 match, that is a valid result and must be
reported as such** -- it means the kernel-launch/sync path is cheap
enough in both languages that neither host adds a measurable tax on
top of GPU time, which is itself useful evidence (it says the earlier,
GPU-bound fusion work from Day 7 dominates total latency regardless of
host language). The experiment is designed so "no host effect" and
"host effect only in jitter, not p50/p99" are both legible, reportable
outcomes -- this package does not assume Rust wins on every axis
before running it.

## Files

- `shim.cu` - the fused kernel (`bilinear_sample` +
  `fused_resize_normalize_chw`, copied exactly from
  `../day07/preprocess.cu`) plus a small `extern "C"` API
  (`fused_init` / `fused_upload` / `fused_run` / `fused_download` /
  `fused_free`) so both hosts can drive it identically. Compiled into
  `libfused.so`.
- `host_py.py` - Python host: ctypes bindings, synthetic rolling
  frame, both experiments, prints `RESULT_JSON: {...}`.
- `host_rs/` - Rust host, same experiments:
  - `Cargo.toml` - no dependencies, `edition = "2021"`,
    `[profile.release] opt-level = 3`.
  - `build.rs` - links `libfused` (from `$FUSED_DIR`) and `cudart`
    (from `/usr/local/cuda/lib64`).
  - `src/main.rs` - the host logic, std only, no unsafe beyond the
    four FFI calls.
- `parity.py` - numerical correctness check: runs the shim on frame 0
  of the rolling pattern (identical to Day 7's static synthetic input)
  and compares against Day 7's reference implementation
  (`../day07/parity.py`) if present, else an inline fallback with the
  same formula.
- `run.sh` - builds everything, runs both hosts, combines their
  `RESULT_JSON` lines into `results.json`, reports Rust binary size,
  checks `ldd` for libpython (must be absent) and libfused/libcudart
  (must be present), and runs `parity.py`.
- `README.md` - this file.

## Setup (Jetson, no sudo)

Install Rust via rustup (one-line, non-interactive, minimal profile):

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source ~/.cargo/env
rustc --version   # should report a stable toolchain
cargo --version
```

`source ~/.cargo/env` (or add it to `~/.bashrc`) is needed in every new
shell that runs `run.sh`, since `rustup` installs to `~/.cargo` with no
sudo and nothing else puts `cargo`/`rustc` on `PATH`.

## Build and run

```
./run.sh
```

or manually:

```
nvcc -O3 -arch=sm_87 -Xcompiler -fPIC -shared -o libfused.so shim.cu

FUSED_DIR="$PWD" cargo build --release --manifest-path host_rs/Cargo.toml

export LD_LIBRARY_PATH="$PWD:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

python3 host_py.py
./host_rs/target/release/host_rs

python3 parity.py
```

`FUSED_DIR` must point at the directory containing `libfused.so` (it
is read by `host_rs/build.rs` at build time, to emit the
`-L` search path for `-lfused`). `LD_LIBRARY_PATH` must include both
that same directory (for `libfused.so`) and `/usr/local/cuda/lib64`
(for `libcudart.so`) at *run* time for either host to find the shared
libraries -- Python's ctypes and Rust's dynamic linker both need this,
build-time linking alone is not enough.

## Problem definition

Identical to Day 7 (`../day07/README.md`, "Problem definition" and
"Synthetic input pattern" and "Bilinear convention" sections apply
unchanged): 1280x720 uint8 HWC in, 3x480x640 float32 CHW out,
half-pixel-center clamped bilinear resize, `x_norm = (x/255 - mean) /
std` with ImageNet mean/std, applied by the exact same kernel code.

The one addition for Day 9 is the **rolling** synthetic input, needed
because both experiments run 1000+600 frames and a perfectly static
input would let a sufficiently aggressive host or kernel cache/skip
work in a way real video input never would:

```
value(frame, y, x, c) = (x*7 + y*13 + c*29 + frame) mod 256
```

`frame` is the 0-based index into the sequence of frames each
experiment runs (tight loop: 0..999; paced loop continues the count
from 1000..1599, so the two experiments never repeat a frame). At
`frame = 0` this is identical to Day 7's static pattern, which is what
makes `parity.py`'s single-frame check comparable directly against Day
7's reference functions with no changes to the reference math.

## Benchmark methodology

- Both hosts call `fused_init` once, then loop `fused_upload` +
  `fused_run` (which itself blocks on `cudaStreamSynchronize`, so each
  call is a complete, GPU-finished frame), then `fused_free` once at
  the end.
- Frame construction (`make_frame` / `build_base_pattern` +
  add-and-mod) happens **outside** every timed region in both hosts --
  only `upload+run` is timed, so the reported numbers isolate the
  host-to-GPU dispatch/sync path, not numpy/Vec array-building cost.
- **Tight loop**: 1000 iterations, first 50 discarded as warmup, then
  mean/p50/p99/max of the remaining 950 per-iteration times, in
  microseconds.
- **Paced loop**: 600 ticks at a 30 Hz target cadence. Jitter =
  `actual_tick_start - target_tick_time`, in microseconds; reported as
  mean/p50/p99/max plus a count of ticks with jitter > 2000 us. Also
  reports `work_mean_us`, the mean upload+run time during the paced
  loop, as a cross-check against the tight loop's numbers under a
  different calling pattern.
- `startup_ms`: wall time from as close to process start as each
  language can measure (`time.perf_counter()` at the top of
  `host_py.py`, before any other import; `Instant::now()` at the top
  of `main()` in `host_rs`) to the first completed `fused_run()` call.
  This captures interpreter/runtime startup, dynamic library loading,
  CUDA context creation (inside `fused_init`), and the first kernel's
  actual execution -- everything between "the OS started this process"
  and "the first frame's GPU work is done."
- Both hosts print one line prefixed `RESULT_JSON: ` with the schema
  below, so `run.sh` can extract and combine them into `results.json`
  without scraping human-readable text.
- The timed region in both hosts is exactly "call `fused_upload`, call
  `fused_run`" -- nothing else. In particular, the `!= 0` error checks
  on each call's return code happen *after* the clock stops in both
  `host_py.py` and `host_rs/src/main.rs` (return codes are stashed in
  local variables and checked once the timing window has closed). This
  was deliberately verified to be symmetric: Python evaluating an `if`
  branch inside the timed region while Rust did not would have handed
  Rust a smaller window than Python's for the same work, biasing the
  one number this experiment is built to report honestly.

### RESULT_JSON schema (both hosts print this)

```json
{
  "host": "python" | "rust",
  "startup_ms": <float>,
  "tight_loop": {
    "iters": 1000, "warmup_discarded": 50,
    "mean_us": <float>, "p50_us": <float>, "p99_us": <float>, "max_us": <float>
  },
  "paced_loop": {
    "ticks": 600, "hz": 30.0,
    "jitter_p50_us": <float>, "jitter_p99_us": <float>, "jitter_max_us": <float>,
    "late_ticks_over_2ms": <int>, "work_mean_us": <float>
  }
}
```

`run.sh` writes `results.json` as `{"python": {...}, "rust": {...}}`.

## Other things run.sh checks

- **Rust binary size**: `stat` on
  `host_rs/target/release/host_rs`, reported in bytes.
- **`ldd` on the Rust binary**: must show `libfused.so` and
  `libcudart.so` and must **not** show `libpython*` -- if it does,
  something pulled a Python dependency into the Rust binary, which
  would defeat the entire point of a from-scratch Rust host (no GIL,
  no interpreter) and needs investigating before trusting the jitter
  numbers. Note `libcudart.so` may appear as a *transitive* dependency
  rather than a direct one: `main.rs` never calls a cudart symbol
  itself (only `fused_*` symbols from `libfused.so`, which is the thing
  that actually calls into CUDA), so a linker that drops unused direct
  `-lcudart` references (e.g. GNU ld with `--as-needed`, common on
  many distros) may not record it as a direct NEEDED entry on the Rust
  binary. `ldd` still resolves and prints transitive dependencies, so
  it will show up via `libfused.so`'s own dependency on it either way
  -- the check is still meaningful, just note it is not necessarily
  evidence the Rust binary itself was linked straight against cudart.
- **Parity**: `parity.py` runs the shim once on frame 0 and diffs
  against Day 7's reference (max abs diff, tolerance `1e-2`, same as
  Day 7).

## How to read the numbers

- **Tight-loop p50/p99, Python vs Rust**: expected to be close. Both
  are GPU-bound (blocking sync per call); a small, consistent Rust
  advantage here would point to residual per-call ctypes overhead, a
  large one would be surprising and worth double-checking (e.g. is the
  Python side accidentally doing extra work per iteration). Report
  whichever result actually comes out -- matching p50/p99 is not a
  failure of the experiment, it is the "fusion dominates, host
  language doesn't" outcome and is exactly as reportable as a gap
  would be.
- **Paced-loop jitter p50/p99/max, Python vs Rust**: this is where the
  claim under test predicts a real difference. If Rust's jitter is
  tighter (lower p99, fewer `late_ticks_over_2ms`), that supports "no
  GIL / no hidden allocator" as the mechanism, independent of whatever
  the tight-loop numbers show. But if *both* hosts show similar jitter
  and it is large relative to the 33 ms tick period (a high p50, a
  large fraction of ticks over 2 ms late), that is a different finding
  than "no host-language effect": it means OS scheduling granularity on
  the machine under test is the dominant source of lateness for both
  hosts, swamping whatever smaller effect the host language contributes
  -- the experiment did not have the resolution to see a host-language
  difference, rather than having established there isn't one. Before
  concluding "no effect" from a result like that, check the Jetson's
  power/clock mode (`nvpmodel -q`, `jetson_clocks` status) and whether
  anything else on the box is loading the CPU during the run; a
  contended or power-throttled Jetson can produce exactly this
  symmetric-large-jitter signature on its own. (This distinction is not
  hypothetical: a non-Jetson sanity run of this harness against a stub
  library, on a loaded/shared development machine, produced ~7 ms
  median jitter for *both* hosts -- clearly scheduler-noise-dominated,
  not a measurement of either language's own precision. That run only
  confirmed the harness's logic and JSON schema; it says nothing about
  what the Jetson will show.)
- **`late_ticks_over_2ms`**: a coarser, single-number version of the
  jitter distribution -- useful for an at-a-glance comparison and for
  noticing whether lateness is many-small or few-large (a high p99
  with a low late-tick count means occasional bad misses; a moderate
  p99 with a high late-tick count means chronic small lateness).
- **`startup_ms`**: expected to favor Rust by a wide margin (no
  interpreter start, no numpy import, no ctypes symbol resolution
  overhead beyond the dynamic linker's own work) -- this is a one-time
  cost, separate from per-frame jitter, and matters most for
  cold-start-sensitive deployments (e.g. a systemd unit restarting
  after a crash) rather than for a long-running service.
- **Binary size**: expected to be small (a static-ish Rust binary with
  no crates beyond std, dynamically linked only to libc/libfused/
  libcudart) -- report it as a data point about deployment footprint,
  not as evidence for or against the jitter claim.
- **libpython absence**: a correctness check on the experiment's
  premise, not a performance number -- if this ever comes back
  non-empty, the Rust binary is not actually running independently of
  Python and the jitter comparison is invalid until that is fixed.

## Results (fill in after running on the Jetson)

Run `./run.sh` on the Jetson Orin Nano and copy the numbers from
`results.json` / the console output here.

| Metric | Python | Rust |
|---|---|---|
| startup_ms | | |
| tight_loop mean_us | | |
| tight_loop p50_us | | |
| tight_loop p99_us | | |
| tight_loop max_us | | |
| paced_loop jitter p50_us | | |
| paced_loop jitter p99_us | | |
| paced_loop jitter max_us | | |
| paced_loop late_ticks_over_2ms | | |
| paced_loop work_mean_us | | |

| Metric | Value |
|---|---|
| Rust binary size (bytes) | |
| `ldd` shows libpython? (must be No) | |
| `ldd` shows libfused? | |
| `ldd` shows libcudart? | |

Parity check result: (paste `parity.py`'s max/mean abs diff and
PASS/FAIL line here)

**Verdict** (fill in once the numbers are in): did tight-loop p50/p99
match between hosts, and did paced-loop jitter differ? State the
result as measured, including a "they matched" outcome if that is what
happened -- see "The claim under test" above for why that is still a
valid, reportable finding.
