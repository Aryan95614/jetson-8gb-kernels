#!/usr/bin/env python3
"""Parse trtexec stdout/stderr text into mean/median/p99 GPU compute time.

Usable as a CLI (`python3 parse_trtexec.py log.txt [--cuda-graph]`) or
imported (`from parse_trtexec import parse_log`).
"""
import json
import re
import sys

_THROUGHPUT_RE = re.compile(r"Throughput:\s*([\d.]+)\s*qps")
# A "[W]" line mentioning "graph" alone is not evidence of a failed capture --
# TensorRT/trtexec emit benign [W] lines (e.g. about ignored flags, or
# unrelated to CUDA graphs at all) that happen to contain the word "graph".
# Require a failure/fallback word alongside it, in either order.
_GRAPH_WARN_RE = re.compile(
    r"\[W\].*graph.*(fail|captur|fallback|ignor|disabl)"
    r"|\[W\].*(fail|captur|fallback|ignor|disabl).*graph",
    re.IGNORECASE,
)


def _parse_stats(text: str, label: str):
    """Match trtexec's "<label>: min = X ms, max = ... percentile(99%) = V ms"."""
    pattern = re.compile(
        rf"{re.escape(label)}:\s*min = ([\d.]+) ms, max = ([\d.]+) ms, "
        rf"mean = ([\d.]+) ms, median = ([\d.]+) ms, "
        rf"percentile\(99%\) = ([\d.]+) ms"
    )
    m = pattern.search(text)
    if not m:
        return None
    keys = ["min", "max", "mean", "median", "p99"]
    return {k: float(v) for k, v in zip(keys, m.groups())}


def parse_log(text: str, used_cuda_graph: bool = False) -> dict:
    gpu_compute = _parse_stats(text, "GPU Compute Time")
    latency = _parse_stats(text, "Latency")
    tp_match = _THROUGHPUT_RE.search(text)
    throughput_qps = float(tp_match.group(1)) if tp_match else None

    cuda_graph_capture = None
    cuda_graph_warnings = []
    if used_cuda_graph:
        # trtexec prints a "[W] ... graph ... fail/fallback/..." warning when
        # it can't use a CUDA graph (capture failed, so it fell back to
        # normal execution). No such warning means capture succeeded. Keep
        # the actual matched lines, not just the bool -- a bare False with no
        # evidence is unfalsifiable if the regex is ever wrong again.
        cuda_graph_warnings = [
            line for line in text.splitlines() if _GRAPH_WARN_RE.search(line)
        ]
        cuda_graph_capture = not bool(cuda_graph_warnings)

    return {
        "gpu_compute_ms": gpu_compute,
        "latency_ms": latency,
        "throughput_qps": throughput_qps,
        "cuda_graph_capture": cuda_graph_capture,
        "cuda_graph_warnings": cuda_graph_warnings,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: parse_trtexec.py LOGFILE [--cuda-graph]", file=sys.stderr)
        raise SystemExit(2)
    path = sys.argv[1]
    used_cuda_graph = "--cuda-graph" in sys.argv[2:]
    with open(path) as f:
        text = f.read()
    print(json.dumps(parse_log(text, used_cuda_graph=used_cuda_graph), indent=2))


if __name__ == "__main__":
    main()
