#!/usr/bin/env bash
# Day 8: build FP16 TensorRT engines from both ONNX graphs and benchmark them
# with trtexec, with and without CUDA graphs. Writes results.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
else
    TRTEXEC="$(command -v trtexec)"
fi
echo "using trtexec: $TRTEXEC"

MODELS=(model_only model_with_preprocess)

build_engine() {
    # stdout of this function is captured by the caller: print ONLY the
    # elapsed build time (seconds, float). trtexec's own output goes to $log.
    # trtexec's exit status is checked manually (not via `set -e`) so that on
    # failure we can surface the tail of its log before aborting -- the uint8
    # NHWC input on model_with_preprocess is the part of this graph most
    # likely to be rejected at parse/build time, and a silent abort under
    # `set -e` would hide exactly the message that explains why.
    local name=$1
    local log="build_${name}.log"
    local t0 t1 rc
    t0=$(date +%s.%N)
    set +e
    "$TRTEXEC" --onnx="${name}.onnx" --saveEngine="${name}.plan" --fp16 --skipInference \
        > "$log" 2>&1
    rc=$?
    set -e
    t1=$(date +%s.%N)
    if [ "$rc" -ne 0 ]; then
        echo "trtexec FAILED building ${name} (exit ${rc}); tail of ${log}:" >&2
        tail -20 "$log" >&2
        exit "$rc"
    fi
    python3 -c "print(${t1} - ${t0})"
}

run_bench() {
    local name=$1
    local variant=$2   # "cudagraph" or "plain"
    local extra=$3     # "--useCudaGraph" or ""
    local log="bench_${name}_${variant}.log"
    "$TRTEXEC" --loadEngine="${name}.plan" --fp16 --iterations=200 --warmUp=500 $extra \
        > "$log" 2>&1 || true
}

echo "== building engines ==" >&2
declare -A BUILD_TIME
for name in "${MODELS[@]}"; do
    echo "-- building ${name} --" >&2
    BUILD_TIME[$name]=$(build_engine "$name")
    echo "${name}: build=${BUILD_TIME[$name]}s" >&2
done

echo "== benchmarking engines ==" >&2
for name in "${MODELS[@]}"; do
    echo "-- benchmarking ${name} --" >&2
    run_bench "$name" cudagraph "--useCudaGraph"
    run_bench "$name" plain ""
done

BT_MODEL_ONLY="${BUILD_TIME[model_only]}" \
BT_MODEL_WITH_PREPROCESS="${BUILD_TIME[model_with_preprocess]}" \
python3 - <<'PYEOF'
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from parse_trtexec import parse_log

MODELS = ["model_only", "model_with_preprocess"]
BUILD_TIME_ENV = {
    "model_only": "BT_MODEL_ONLY",
    "model_with_preprocess": "BT_MODEL_WITH_PREPROCESS",
}

results = {}
for name in MODELS:
    plan_path = f"{name}.plan"
    entry = {
        "build_seconds": float(os.environ[BUILD_TIME_ENV[name]]),
        "engine_size_bytes": os.path.getsize(plan_path),
    }
    for variant, used_graph in (("cudagraph", True), ("plain", False)):
        with open(f"bench_{name}_{variant}.log") as f:
            text = f.read()
        entry[variant] = parse_log(text, used_cuda_graph=used_graph)
    results[name] = entry

all_results = {}
if os.path.exists("results.json"):
    with open("results.json") as f:
        all_results = json.load(f)
all_results["build_and_bench"] = results
with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("wrote results.json (build_and_bench)")
PYEOF
