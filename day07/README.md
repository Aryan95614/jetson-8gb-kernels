# Day 7: a fused Ampere preprocess kernel

Target: Jetson Orin Nano 8GB Super (Ampere, sm_87), CUDA 12.6, JetPack 6.2,
L4T R36.4.2, Python 3.10 with torch 2.8 + numpy.

## What this shows

A typical image preprocessing pipeline does three separate steps, each a
separate CUDA kernel writing its result back to DRAM before the next
kernel reads it:

1. resize (bilinear, HWC uint8 -> HWC uint8)
2. normalize (scale to [0,1], subtract mean, divide by std, HWC uint8 -> HWC float32)
3. layout convert (HWC float32 -> CHW float32)

That is three kernel launches and, more importantly, three round trips
through DRAM: each kernel reads its input from global memory and writes
its output back to global memory, and the next kernel reads that back
again. A single fused kernel that does resize + normalize + layout in
one pass reads the source image once and writes the final tensor once,
paying for one DRAM round trip and one launch instead of three of each.

This package builds both versions, times them with the same input on
the same GPU, and reports where the time goes (launch overhead,
achieved bandwidth, occupancy) so the "fusion wins" claim is backed by
numbers, not just intuition.

## Files

- `preprocess.cu` - the four CUDA kernels plus a host benchmark `main()`.
- `parity.py` - numerical correctness check for the fused kernel's output.
- `run.sh` - builds, runs, dumps, checks parity, writes `results.json`.
- `README.md` - this file.

## Build

```
nvcc -O3 -arch=sm_87 -o preprocess preprocess.cu
```

## Run

```
./run.sh
```

or manually:

```
nvcc -O3 -arch=sm_87 -o preprocess preprocess.cu
./preprocess --dump out.bin
python3 parity.py --dump out.bin
```

`./preprocess` alone runs the benchmark and prints results but does not
write `out.bin`; pass `--dump out.bin` to also write the fused kernel's
output tensor to disk (needed by `parity.py`).

If OpenCV is present on the Jetson, it is worth running `parity.py`
once each way and confirming they agree with each other before
trusting either one against `out.bin`:

```
python3 parity.py --dump out.bin           # cv2 backend, if importable
python3 parity.py --dump out.bin --no-cv2  # pure-numpy backend, always available
```

Both were only exercised against each other in a self-test on a
non-Jetson machine without OpenCV installed (see the numpy fallback
path); this two-command check isolates a resize-backend disagreement
from an actual kernel bug the first time both are available together.

## Problem definition

- Input: 1280x720 RGB, uint8, HWC layout (row-major, 3 interleaved
  channel bytes per pixel). This is a synthetic image, not a real
  photo; see "Synthetic input pattern" below.
- Output: 3x480x640, float32, CHW layout (one contiguous plane per
  channel). This is a single frame; add a leading batch dimension of 1
  in the consuming framework if needed, nothing here changes for that.
- Resize: bilinear (`INTER_LINEAR`-equivalent), half-pixel-center
  convention, see below.
- Normalize: `x_norm = (x/255 - mean) / std`, applied per channel,
  channel order unchanged (RGB stays RGB, this package does no color
  conversion).
  - mean = (0.485, 0.456, 0.406)
  - std  = (0.229, 0.224, 0.225)
- Precision: float32 output, not float16. Chosen for simplicity - it
  keeps the arithmetic and the parity check straightforward, and this
  is a bandwidth/launch-overhead demo, not a precision or throughput
  ceiling demo. A float16 output variant would roughly halve the
  output-side bytes and is a natural follow-up, not implemented here.

## Synthetic input pattern

Both `preprocess.cu` (`fill_synthetic_input()`) and `parity.py`
(`build_synthetic_input()`) generate the exact same deterministic
1280x720x3 uint8 image, with no file I/O and no RNG, so the two sides
have something to agree on:

```
value(y, x, c) = (x*7 + y*13 + c*29) mod 256
```

where `x` in [0, 1280), `y` in [0, 720), `c` in {0=R, 1=G, 2=B}.

## Bilinear convention

Both the CUDA kernels and `parity.py`'s reference implementation use
OpenCV's `INTER_LINEAR` convention: half-pixel centers, clamped edges.

```
scale = src_size / dst_size
src_coord = (dst_coord + 0.5) * scale - 0.5
src_coord = clamp(src_coord, 0, src_size - 1)
```

Then the usual 4-tap bilinear blend using `floor(src_coord)` and
`floor(src_coord) + 1` (also clamped to `src_size - 1`) as the two taps
per axis. See `bilinear_sample()` in `preprocess.cu` and
`bilinear_resize_numpy()` in `parity.py` - both implement this formula
directly, and `parity.py`'s optional OpenCV backend calls
`cv2.resize(..., interpolation=cv2.INTER_LINEAR)` on a float32 image
(not uint8; see the comment in `parity.py` for why) to get the same
convention out of OpenCV itself.

## Kernels and fusion

| Kernel | Reads | Writes | Role |
|---|---|---|---|
| `resize_bilinear_hwc_u8` | uint8 HWC, 1280x720 | uint8 HWC, 640x480 | path A, stage 1 |
| `normalize_hwc_f32` | uint8 HWC, 640x480 | float32 HWC, 640x480 | path A, stage 2 |
| `hwc_to_chw_f32` | float32 HWC, 640x480 | float32 CHW, 640x480 | path A, stage 3 |
| `fused_resize_normalize_chw` | uint8 HWC, 1280x720 | float32 CHW, 640x480 | path B, one stage |

Path A = `resize_bilinear_hwc_u8` -> `normalize_hwc_f32` -> `hwc_to_chw_f32`,
three launches, three DRAM round trips.

Path B = `fused_resize_normalize_chw` alone: for each output (channel,
row, 4-wide column group), it bilinear-samples directly from the uint8
source, normalizes, and stores a `float4` straight into the CHW output.
One launch, one DRAM round trip.

Note: because path A's stage 1 physically writes a rounded uint8
intermediate buffer to DRAM, while the fused kernel keeps the
interpolated value in a register as float32 and never rounds it to
uint8, path A's and path B's outputs are expected to differ very
slightly (up to roughly half a uint8 level's worth of normalized
value, well under 1/255). `parity.py` checks path B's output (`out.bin`
is the fused dump) against a reference built the same way (float32
throughout, no intermediate uint8 rounding), so this expected
difference does not show up as a parity failure.

Vectorization notes:
- `resize_bilinear_hwc_u8` reads scattered, non-vectorizable taps (4
  bilinear taps x 3 channels, at arbitrary offsets); no vectorized
  load is used there, this is inherent to gather-style resize.
- `normalize_hwc_f32` treats its HWC buffer as a flat array; 640*480*3
  is divisible by 4, so each thread does one `uchar4` load and one
  `float4` store.
- `hwc_to_chw_f32` and `fused_resize_normalize_chw` both process 4
  consecutive output columns per thread and issue one `float4` store
  into the CHW output (640 is divisible by 4, so this needs no
  remainder handling); their reads stay scalar because the source
  layout is strided (HWC, or scattered bilinear taps) either way.

No shared memory is used anywhere; the working set per block is small
enough that L1/L2 caching handles tap reuse.

## Benchmark methodology

- 50 warmup iterations (both paths), then 1000 timed iterations per
  path, `cudaEvent` timing around the full path (all three kernels for
  path A, the one kernel for path B), synchronized per iteration.
- Reports mean, p50, and p99 microseconds per frame per path.
- Reports achieved GB/s = nominal per-frame bytes moved / mean time.
  "Nominal" bytes = input size + output size at each stage, summed
  across stages for path A. This does not individually count every
  overlapping bilinear tap read (adjacent output pixels reuse source
  bytes, and L1/L2 caching absorbs that reuse in practice), so treat
  the GB/s figures as a comparable-between-paths bandwidth estimate,
  not a cache-miss-exact hardware counter measurement.
- Reports occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)
  and registers per thread (`cudaFuncGetAttributes`) for all four
  kernels, at the block size each is actually launched with.
- Prints one line prefixed `RESULT_JSON: ` with all of the above as
  JSON, so `run.sh` can extract it into `results.json` without
  scraping the human-readable text.
- Each timed iteration synchronizes (`cudaEventSynchronize`) before the
  next one starts, so every iteration launches into an otherwise-idle
  GPU. That is a clean, reproducible measurement, but it also means
  path A's later kernels can enqueue while an earlier one is still
  executing, partially hiding their launch latency behind that
  execution. In a real back-to-back-frames pipeline (no forced sync
  between frames) launch overhead would pipeline differently and path
  A's disadvantage could be larger. Treat the measured path A vs path B
  gap as a lower bound on the benefit of fusion, not an upper bound.

## How to read the numbers

- **Path A mean/p50/p99 vs path B mean/p50/p99**: path B should be
  faster on all three; the gap is the combined cost of two extra
  kernel launches plus the two extra DRAM round trips path A pays for
  (uint8 resize buffer, float32 HWC buffer).
- **Speedup** (`path_a.mean_us / path_b.mean_us`): expect greater than
  1.0; how much greater is the headline number for this experiment.
- **GB/s**: path B's bytes-moved figure is much smaller than path A's
  (no intermediate buffers to write and re-read), so even a modest
  GB/s number for path B can represent less total DRAM traffic than a
  higher GB/s number for path A - compare total time, not GB/s alone,
  when judging "which is faster."
- **Occupancy / registers**: sanity-checks that neither path is
  bottlenecked by register pressure or low occupancy relative to the
  other; if path A's per-kernel occupancy is fine but it still loses,
  that confirms the bottleneck is launch/DRAM overhead between
  kernels, not per-kernel execution efficiency, which is the point of
  this experiment.

## Results (fill in after running on the Jetson)

Run `./run.sh` on the Jetson Orin Nano and copy the numbers from
`results.json` / the console output here.

| Metric | Path A (3 kernels) | Path B (fused) |
|---|---|---|
| mean us/frame | | |
| p50 us/frame | | |
| p99 us/frame | | |
| GB/s | | |
| Speedup (A/B) | | 1.00x (reference) |

| Kernel | regs/thread | block size | max active blocks/SM | occupancy |
|---|---|---|---|---|
| resize_bilinear_hwc_u8 | | | | |
| normalize_hwc_f32 | | | | |
| hwc_to_chw_f32 | | | | |
| fused_resize_normalize_chw | | | | |

Parity check result: (paste `parity.py`'s max/mean abs diff and
PASS/FAIL line here)
