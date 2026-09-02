# jetson-8gb-kernels

Kernels and hosts from "Day X/30 of inference engineering on an 8 GB NVIDIA Jetson".
Board: Jetson Orin Nano 8GB Super dev kit, JetPack 6.2 (L4T R36.4.2), CUDA 12.6, sm_87, 15W mode unless stated.
Every number in the tables below was produced on that board by the script in the same directory. Rerun it; if your number differs, open an issue with your mode line.

| dir | what | measured (15W) | parity |
|---|---|---|---|
| day07 | fused preprocess kernel: 1280x720 u8 HWC -> 3x480x640 f32 CHW, bilinear + normalize | 3 kernels 479.9 us/frame, fused 205.5 us/frame, 2.34x; 28 regs/thread, occupancy 1.00 | max abs diff 0.000000 vs OpenCV INTER_LINEAR |
| day08 | TensorRT FP16 engines, preprocess outside vs folded into the graph | (pending) | logits diff (pending) |
| day09 | same kernel, Python host vs Rust host: p50/p99 and 30 Hz jitter | (pending) | (pending) |

Build: `cd day07 && nvcc -O3 -arch=sm_87 -o preprocess preprocess.cu && bash run.sh`
