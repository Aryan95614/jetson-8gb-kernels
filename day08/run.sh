#!/usr/bin/env bash
# Day 8: full pipeline. Export ONNX -> build+bench TensorRT engines -> true
# end-to-end latency. Run this ON THE JETSON (JetPack 6.2 / TensorRT 10.3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== step 1/3: export ONNX =="
python3 export_onnx.py

echo "== step 2/3: build + bench TensorRT engines (trtexec) =="
bash build.sh

echo "== step 3/3: true end-to-end latency (tensorrt python API) =="
python3 e2e.py

echo
echo "== summary (results.json) =="
python3 - <<'PYEOF'
import json

with open("results.json") as f:
    r = json.load(f)

bb = r.get("build_and_bench", {})
print(f"{'engine':24s} {'size(MB)':>10s} {'plain mean(ms)':>16s} {'cudagraph mean(ms)':>20s} {'graph captured':>15s}")
for name, entry in bb.items():
    size_mb = entry["engine_size_bytes"] / (1024 * 1024)
    plain = (entry.get("plain") or {}).get("gpu_compute_ms") or {}
    graph = (entry.get("cudagraph") or {}).get("gpu_compute_ms") or {}
    captured = (entry.get("cudagraph") or {}).get("cuda_graph_capture")
    print(
        f"{name:24s} {size_mb:10.1f} {plain.get('mean', float('nan')):16.3f} "
        f"{graph.get('mean', float('nan')):20.3f} {str(captured):>15s}"
    )

e2e = r.get("e2e", {})
print()
print(f"{'path':24s} {'gpu mean(ms)':>13s} {'gpu p99(ms)':>12s} {'wall/frame(ms)':>15s}")
for name, stats in e2e.items():
    if not isinstance(stats, dict) or "gpu_ms" not in stats:
        continue
    gpu = stats["gpu_ms"]
    print(
        f"{name:24s} {gpu['mean']:13.3f} {gpu['p99']:12.3f} "
        f"{stats['wall_ms_per_frame']:15.3f}"
    )
if "max_abs_logit_diff_a_vs_b" in e2e:
    print(f"\nmax abs logit diff (A vs B): {e2e['max_abs_logit_diff_a_vs_b']:.4f}")
print(
    "\nnote: wall/frame (not gpu mean) is the number that sets framerate -- "
    "gpu mean only counts kernel-busy time and misses CPU/PyTorch dispatch "
    "gaps between async-enqueued iterations."
)
PYEOF
