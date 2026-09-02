#!/usr/bin/env bash
# run.sh - Day 9: Rust host vs Python host launching the same fused CUDA
# kernel. Builds libfused.so (nvcc), builds host_rs (cargo, release),
# runs host_py.py and host_rs, combines their RESULT_JSON lines into
# results.json, reports the Rust binary's size and linked libraries
# (must NOT link libpython), and runs the parity check.
#
# Usage: ./run.sh
# Expected environment: Jetson Orin Nano 8GB Super, JetPack 6.2,
# CUDA 12.6, nvcc on PATH, python3 with numpy, Rust stable via rustup
# (source ~/.cargo/env first -- see README.md).
#
# Produces:
#   libfused.so             - shared library built from shim.cu
#   host_rs/target/release/host_rs - the Rust host binary
#   results.json             - {"python": {...}, "rust": {...}}

set -eu
set -o pipefail

cd "$(dirname "$0")"
HERE="$PWD"

echo "== Building libfused.so (nvcc -O3 -arch=sm_87 -Xcompiler -fPIC -shared) =="
nvcc -O3 -arch=sm_87 -Xcompiler -fPIC -shared -o libfused.so shim.cu

echo "== Building host_rs (cargo build --release) =="
export FUSED_DIR="$HERE"
cargo build --release --manifest-path host_rs/Cargo.toml

RUST_BIN="$HERE/host_rs/target/release/host_rs"
export LD_LIBRARY_PATH="$HERE:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== Running host_py.py (Python host: 1000-iter tight loop + 600-tick paced loop) =="
PY_OUT="$(python3 host_py.py)"
echo "$PY_OUT"
PY_LINE="$(printf '%s\n' "$PY_OUT" | grep '^RESULT_JSON: ' || true)"
if [ -z "$PY_LINE" ]; then
    echo "run.sh: ERROR: no RESULT_JSON line found in host_py.py output" >&2
    exit 1
fi

echo "== Running host_rs (Rust host: same two experiments) =="
RS_OUT="$("$RUST_BIN")"
echo "$RS_OUT"
RS_LINE="$(printf '%s\n' "$RS_OUT" | grep '^RESULT_JSON: ' || true)"
if [ -z "$RS_LINE" ]; then
    echo "run.sh: ERROR: no RESULT_JSON line found in host_rs output" >&2
    exit 1
fi

python3 - "$PY_LINE" "$RS_LINE" > results.json <<'PYEOF'
import json
import sys

py_json = sys.argv[1][len("RESULT_JSON: "):]
rs_json = sys.argv[2][len("RESULT_JSON: "):]
combined = {"python": json.loads(py_json), "rust": json.loads(rs_json)}
print(json.dumps(combined, indent=2))
PYEOF

echo "== Wrote results.json =="
cat results.json

echo "== Rust binary size =="
RS_SIZE_BYTES="$(stat -c%s "$RUST_BIN" 2>/dev/null || stat -f%z "$RUST_BIN")"
echo "rust_binary_bytes: $RS_SIZE_BYTES"

echo "== ldd $RUST_BIN =="
LDD_OUT="$(ldd "$RUST_BIN" 2>&1 || true)"
echo "$LDD_OUT"

LIBPYTHON_LINE="$(printf '%s\n' "$LDD_OUT" | grep -i libpython || true)"
if [ -n "$LIBPYTHON_LINE" ]; then
    echo "run.sh: WARNING: rust binary links libpython (it must not):" >&2
    echo "$LIBPYTHON_LINE" >&2
else
    echo "libpython check: OK, not linked"
fi

LIBFUSED_LINE="$(printf '%s\n' "$LDD_OUT" | grep -i libfused || true)"
LIBCUDART_LINE="$(printf '%s\n' "$LDD_OUT" | grep -i libcudart || true)"
echo "libfused link:  ${LIBFUSED_LINE:-NOT FOUND}"
echo "libcudart link: ${LIBCUDART_LINE:-NOT FOUND}"

echo "== Parity check (parity.py) =="
set +e
python3 parity.py
PARITY_STATUS=$?
set -e

if [ "$PARITY_STATUS" -ne 0 ]; then
    echo "run.sh: parity check FAILED (exit $PARITY_STATUS)" >&2
    exit "$PARITY_STATUS"
fi

echo "== Done. See results.json. =="
