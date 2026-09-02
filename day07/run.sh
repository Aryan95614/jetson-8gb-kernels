#!/usr/bin/env bash
# run.sh - Day 7 fused preprocess kernel: build, benchmark, dump, verify.
#
# Usage: ./run.sh
# Expected environment: Jetson Orin Nano 8GB Super, JetPack 6.2,
# CUDA 12.6, nvcc on PATH, python3 with numpy (cv2 optional).
#
# Produces:
#   preprocess   - compiled benchmark binary
#   out.bin      - fused kernel output dump (float32 CHW), for parity.py
#   results.json - the benchmark's RESULT_JSON line, as a standalone file

set -eu
set -o pipefail

cd "$(dirname "$0")"

echo "== Building preprocess (nvcc -O3 -arch=sm_87) =="
nvcc -O3 -arch=sm_87 -o preprocess preprocess.cu

echo "== Running benchmark (50 warmup + 1000 timed iters per path) =="
BENCH_OUT="$(./preprocess --dump out.bin)"
echo "$BENCH_OUT"

RESULT_LINE="$(printf '%s\n' "$BENCH_OUT" | grep '^RESULT_JSON: ' || true)"
if [ -z "$RESULT_LINE" ]; then
    echo "run.sh: ERROR: no RESULT_JSON line found in benchmark output" >&2
    exit 1
fi
printf '%s\n' "${RESULT_LINE#RESULT_JSON: }" > results.json
echo "== Wrote results.json =="
cat results.json

echo "== Running parity check (parity.py) =="
set +e
python3 parity.py --dump out.bin
PARITY_STATUS=$?
set -e

if [ "$PARITY_STATUS" -ne 0 ]; then
    echo "run.sh: parity check FAILED (exit $PARITY_STATUS)" >&2
    exit "$PARITY_STATUS"
fi

echo "== Done. See results.json and out.bin. =="
