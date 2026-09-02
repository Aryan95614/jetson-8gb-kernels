# Day 8: TensorRT plugins, when the graph has a hole

**Honest framing up front:** this is NOT a custom TensorRT plugin. We do not
write any C++, register any `IPluginV3`, or touch the plugin registry. We fold
preprocessing into the ONNX graph using stock ONNX ops (cast, permute, resize,
sub, div) so that TensorRT owns the whole camera-to-logits path. Call it the
"ONNX-op version," not "a custom plugin." A real plugin would be the move if
the op TensorRT can't express natively (e.g. a fused Bayer-demosaic + resize
kernel); here, `Resize`/`Cast`/`Transpose` are all things TensorRT already
imports and executes, so there's no gap a plugin needs to fill.

## the point

If your preprocessing (resize/normalize/layout conversion) happens in Python
between the camera and `context.execute_*`, you are not running TensorRT
end-to-end -- you're running TensorRT for the matmuls and eating a Python/
PyTorch round trip for everything else. That round trip does not show up in
`trtexec`'s "GPU Compute Time," because `trtexec` only ever sees the engine.
It shows up in `t_e2e`, which is what actually determines your framerate.

This day builds two engines from the same resnet18 (random weights -- this is
a latency/plumbing test, not an accuracy test):

- `model_only.plan`: takes an already-preprocessed float NCHW tensor. All
  preprocessing lives in `e2e.py`, in PyTorch, outside the engine.
- `model_with_preprocess.plan`: takes the raw uint8 NHWC camera frame. Cast,
  permute, resize, and normalize are all ops inside the ONNX graph, so
  TensorRT builds, fuses, and schedules them along with the backbone.

Then `e2e.py` measures `t_e2e` for both, with the *same synthetic input data*,
so the only thing that differs is where the preprocessing ops live.

## how to read the numbers

Each `e2e.py` path reports two numbers, and they answer different questions:

- `gpu_ms` (from `torch.cuda.Event` pairs around each iteration): GPU-busy
  time for that one iteration's kernels only. Because all 1000 iterations are
  enqueued asynchronously on a single stream, CPU-side Python/PyTorch
  dispatch overhead lands in the *gap between* one iteration's end event and
  the next iteration's start event -- it never lands inside a measured
  interval, even when dispatch is the actual bottleneck. `gpu_ms` is close to
  what `trtexec`'s own `GPU Compute Time` measures.
- `wall_ms_per_frame` (wall-clock around the whole loop, divided by n): real
  per-frame throughput, including CPU dispatch, kernel launch latency, and
  any queue backlog. **This is the number that sets your framerate**, and the
  one to use for the A-vs-B comparison below.

Numbers:

- `results.json["build_and_bench"][engine]["plain"|"cudagraph"]["gpu_compute_ms"]`
  is `trtexec`'s own engine-only number -- this is what people usually quote,
  and it is *not* `t_e2e`. It won't show the hole.
- `results.json["e2e"]["inside_gpu_frame"]` (path B) is the closest thing to
  a true `t_e2e` for `model_with_preprocess.plan`: raw frame in, logits out,
  nothing else -- a single `execute_async_v3` call, no separate PyTorch ops.
- `results.json["e2e"]["outside_gpu_frame"]` (path A) is `t_e2e` for the split
  design with the frame already resident on the GPU. A has several extra
  PyTorch op launches per frame (cast, permute, interpolate, divide,
  subtract, divide) that B doesn't -- B is one `execute_async_v3` call. Where
  that cost shows up depends on whether the pipeline is dispatch-bound or
  GPU-bound, so look at `wall_ms_per_frame - gpu_ms.mean` (per-frame time
  NOT spent running kernels) for A vs. B, not just the raw means:
    - If that difference is clearly larger for A/A2 than for B: the pipeline
      is dispatch-bound and the hole shows up in `wall_ms_per_frame` directly
      -- CPU dispatch was outrunning the GPU, so extra Python/PyTorch calls
      cost real wall-clock time.
    - If that difference is small and similar across A/A2/B: the pipeline is
      GPU-bound (a resnet18 forward pass dwarfs a handful of op launches), so
      async dispatch hides the CPU cost behind GPU work and `wall_ms_per_frame`
      for A and B may land close together. In that case look at `gpu_ms.mean`
      instead: A's unfused resize/normalize/cast kernels running as separate,
      unfused CUDA launches should still cost more GPU time than B's fused
      graph, even though the wall-clock gap is small. Don't assume in advance
      which regime this lands in -- read both numbers and report which one
      actually moved.
- `results.json["e2e"]["outside_cpu_frame"]` (path A2) adds a pinned-memory
  H2D copy on top of A, to show what a real camera source (frame arrives on
  the CPU) actually costs relative to a frame that was magically already on
  the GPU.
- `max_abs_logit_diff_a_vs_b`: A and B should produce nearly the same logits
  for the same input frame. They won't match exactly -- FP16 accumulation
  order differs, and PyTorch's `interpolate` and TensorRT's `Resize` are not
  bit-identical implementations of "bilinear." A diff under ~1e-1 on the
  logits of a random-weight resnet18 is expected and fine; this is a plumbing
  check (same graph, same answer), not an accuracy benchmark.
- `cuda_graph_capture` (under each engine's `"cudagraph"` entry) tells you
  whether `trtexec --useCudaGraph` actually captured a graph or silently fell
  back (it prints a `[W]` warning on fallback, which is what we grep for).

## files

- `export_onnx.py` -- builds both ONNX graphs from a random-weight
  torchvision resnet18, runs `onnx.checker.check_model`, prints I/O contracts.
- `build.sh` -- builds both FP16 `.plan` files with `trtexec`, times the
  builds, records engine sizes, benchmarks each engine with and without
  `--useCudaGraph`, writes `results.json`.
- `parse_trtexec.py` -- tiny regex parser for `trtexec` stdout; extracts
  mean/median/p99 GPU compute time, latency, throughput, and CUDA-graph
  capture status. CLI (`python3 parse_trtexec.py log.txt`) or `import`.
- `e2e.py` -- true end-to-end latency using the TensorRT 10.x Python API
  (`set_tensor_address` + `execute_async_v3` on a `torch.cuda.Stream`; no
  `execute_v2`/bindings-list calls). Measures paths A / A2 / B over 1000
  synthetic frames (first 50 discarded as warmup), merges results into
  `results.json`, and prints a `RESULT_JSON:` line.
- `contract.md` -- the input contract table for both engines (layout, dtype,
  shape, color order, mean/std, resize method) plus what lives inside each
  engine vs. outside it.
- `run.sh` -- runs the whole pipeline end to end and prints a summary table.
- `results.json` -- produced by `build.sh` and `e2e.py`; not checked in ahead
  of a run.

## exact commands (run on the Jetson, not here)

```bash
cd ~/remarkable-claude/daily-posts/kernels/day08   # or wherever this landed on-device
python3 export_onnx.py
bash build.sh
python3 e2e.py
# or just:
bash run.sh
```

`build.sh` looks for `trtexec` at `/usr/src/tensorrt/bin/trtexec` first (the
standard JetPack location) and falls back to `$(command -v trtexec)`.

## environment this was written for

Jetson Orin Nano 8GB Super, JetPack 6.2 / L4T R36.4.2, CUDA 12.6, TensorRT
10.3 (Python bindings `import tensorrt`, CLI at
`/usr/src/tensorrt/bin/trtexec`), torch 2.8 + torchvision 0.23, onnx 1.22,
onnxscript 0.7, numpy 1.26, opencv 4.8, Python 3.10. No sudo, no internet
assumed beyond packages already installed via pip.

Executed on that board at 15W; the numbers in the table above and in
`results.jetson.json` are from that run. Rerun `bash run.sh` on your own Orin
Nano and diff your `results.json` against it.
