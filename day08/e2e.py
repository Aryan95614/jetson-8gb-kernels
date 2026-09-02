#!/usr/bin/env python3
"""Day 8: measure true end-to-end latency for two placements of preprocessing.

  A  "outside_gpu_frame" : frame already on GPU -> torch preprocess -> model_only.plan
  A2 "outside_cpu_frame" : frame on pinned CPU memory -> H2D copy -> torch preprocess -> model_only.plan
  B  "inside_gpu_frame"  : frame already on GPU, straight into model_with_preprocess.plan

Uses the TensorRT 10.x tensor-name API (set_tensor_address + execute_async_v3
on a torch CUDA stream); no execute_v2 / bindings-list calls anywhere.

Frame content ("rolling by frame index") is precomputed into a small ring of
distinct frames BEFORE timing starts, so the timed region is only the
pipeline stages under test, not synthetic-frame generation. See README.md
for how to read gpu_ms vs wall_ms_per_frame.
"""
import json
import os
import time

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

N_FRAMES = 1000
WARMUP_DISCARD = 50
RING = 64  # distinct rolled frames, cycled through during timing
H, W = 720, 1280
OUT_H, OUT_W = 480, 640
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.BOOL: torch.bool,
}


class TrtRunner:
    """Wraps one engine with one input tensor name, one output tensor name."""

    def __init__(self, plan_path: str, input_name: str, output_name: str):
        # self.runtime, not a local: a GC'd runtime with a live engine/context
        # is a known source of segfaults / invalid-handle errors in TRT's
        # Python bindings.
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(plan_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"{plan_path}: deserialize_cuda_engine returned None")
        self.context = self.engine.create_execution_context()
        self.input_name = input_name
        self.output_name = output_name

        io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        missing = [n for n in (input_name, output_name) if n not in io_names]
        if missing:
            raise RuntimeError(f"{plan_path}: missing tensor(s) {missing!r}; has {io_names!r}")

        in_shape = tuple(self.engine.get_tensor_shape(input_name))
        if -1 in in_shape:
            raise RuntimeError(f"{plan_path}: dynamic input shape not supported here")

        out_shape = tuple(self.engine.get_tensor_shape(output_name))
        out_dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(output_name)]
        self.output = torch.empty(out_shape, dtype=out_dtype, device="cuda")
        self.context.set_tensor_address(output_name, self.output.data_ptr())

    def infer(self, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.execute_async_v3(stream.cuda_stream)
        return self.output


def make_base_frame() -> torch.Tensor:
    """HWC uint8 pattern (x*7 + y*13 + c*29) % 256, on CPU."""
    x = torch.arange(W).view(1, W, 1)
    y = torch.arange(H).view(H, 1, 1)
    c = torch.arange(3).view(1, 1, 3)
    return ((x * 7 + y * 13 + c * 29) % 256).to(torch.uint8)


def build_frame_bank():
    base = make_base_frame()
    gpu_frames, cpu_frames = [], []
    for i in range(RING):
        rolled = torch.roll(base, shifts=i, dims=0)
        gpu_frames.append(rolled.to("cuda").unsqueeze(0))
        cpu_frames.append(rolled.pin_memory().unsqueeze(0))
    return gpu_frames, cpu_frames


def preprocess_gpu(frame_nhwc_u8: torch.Tensor) -> torch.Tensor:
    x = frame_nhwc_u8.float()
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(
        x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False, antialias=False
    )
    x = x / 255.0
    x = (x - MEAN) / STD
    return x.contiguous()


def summarize(times_ms):
    arr = np.array(times_ms[WARMUP_DISCARD:])
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


def timed_run(name, fn, n=N_FRAMES):
    """Time n iterations of fn(i, stream), two ways, on the same loop.

    gpu_ms: torch.cuda.Event pairs around each iteration -- GPU-busy time for
    that iteration's kernels ONLY. Iterations are enqueued asynchronously on
    one stream, so CPU-side dispatch overhead falls in the gap *between* one
    end event and the next start event, invisible to gpu_ms even when it is
    the real bottleneck.

    wall_ms_per_frame: one wall-clock bracket (synchronized at both ends)
    around the whole loop, divided by n -- real per-frame throughput,
    including CPU dispatch, launch latency, and queue backlog. This is the
    number that sets your framerate; see README.md.
    """
    stream = torch.cuda.Stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for i in range(n):
            starts[i].record(stream)
            fn(i, stream)
            ends[i].record(stream)
    torch.cuda.synchronize()
    wall_t1 = time.perf_counter()

    gpu_times_ms = [starts[i].elapsed_time(ends[i]) for i in range(n)]
    gpu_stats = summarize(gpu_times_ms)
    wall_ms_per_frame = (wall_t1 - wall_t0) * 1e3 / n
    stats = {"gpu_ms": gpu_stats, "wall_ms_per_frame": wall_ms_per_frame}
    print(
        f"{name:24s} gpu_mean={gpu_stats['mean']:.3f}ms "
        f"gpu_p99={gpu_stats['p99']:.3f}ms wall/frame={wall_ms_per_frame:.3f}ms"
    )
    return stats


def main():
    gpu_frames, cpu_frames = build_frame_bank()

    model_only = TrtRunner("model_only.plan", "tensor", "logits")
    model_pre = TrtRunner("model_with_preprocess.plan", "frame", "logits")

    def run_a(i, stream):
        x = preprocess_gpu(gpu_frames[i % RING])
        model_only.infer(x, stream)

    def run_b(i, stream):
        model_pre.infer(gpu_frames[i % RING], stream)

    def run_a2(i, stream):
        frame_gpu = cpu_frames[i % RING].to("cuda", non_blocking=True)
        x = preprocess_gpu(frame_gpu)
        model_only.infer(x, stream)

    # Correctness check: same frame through A and B, compare logits.
    check_stream = torch.cuda.Stream()
    with torch.cuda.stream(check_stream):
        frame0 = gpu_frames[0]
        out_a = model_only.infer(preprocess_gpu(frame0), check_stream).clone()
        out_b = model_pre.infer(frame0, check_stream).clone()
    torch.cuda.synchronize()
    max_abs_diff = float((out_a - out_b).abs().max().item())
    print(f"max abs diff (A vs B logits) = {max_abs_diff:.4f} "
          "(nonzero expected: fp16 + resize-impl differences; <1e-1 is fine)")

    results = {
        "outside_gpu_frame": timed_run("A outside (gpu frame)", run_a),
        "inside_gpu_frame": timed_run("B inside (gpu frame)", run_b),
        "outside_cpu_frame": timed_run("A2 outside (cpu->gpu)", run_a2),
        "max_abs_logit_diff_a_vs_b": max_abs_diff,
    }

    payload = {"e2e": results}
    print("RESULT_JSON:" + json.dumps(payload))

    all_results = {}
    if os.path.exists("results.json"):
        with open("results.json") as f:
            all_results = json.load(f)
    all_results.update(payload)
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
