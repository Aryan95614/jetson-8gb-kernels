// preprocess.cu
// Day 7: fused Ampere preprocess kernel benchmark.
//
// Compares two ways to turn a 1280x720 RGB uint8 (HWC) frame into a
// 3x480x640 float32 (CHW) tensor, normalized with ImageNet mean/std:
//
//   Path A (three kernels, three DRAM round trips):
//     resize_bilinear_hwc_u8  -> normalize_hwc_f32 -> hwc_to_chw_f32
//   Path B (one fused kernel, one DRAM round trip):
//     fused_resize_normalize_chw
//
// Bilinear convention (matches OpenCV cv2.resize with INTER_LINEAR):
//   src_coord = (dst_coord + 0.5) * scale - 0.5, then clamp to [0, N-1]
//   where scale = src_size / dst_size. Edges are clamped, not wrapped.
//
// Build: nvcc -O3 -arch=sm_87 -o preprocess preprocess.cu
// Run:   ./preprocess [--dump out.bin]
//
// This file targets Jetson Orin Nano (Ampere, sm_87), CUDA 12.6.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Fixed problem size (compile-time constants keep the kernels simple and
// let the host code hardcode the nominal byte-traffic figures below).
// ---------------------------------------------------------------------
static const int SRC_W = 1280;
static const int SRC_H = 720;
static const int DST_W = 640;
static const int DST_H = 480;
static const int CH    = 3;

static const int WARMUP_ITERS = 50;
static const int TIMED_ITERS  = 1000;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__constant__ float d_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float d_std[3]  = {0.229f, 0.224f, 0.225f};

// ---------------------------------------------------------------------
// Shared bilinear sampler. src is HWC uint8, W x H x 3.
// Coordinates are pre-computed by the caller using the half-pixel-center
// convention; this function only clamps and interpolates.
// ---------------------------------------------------------------------
__device__ __forceinline__ float bilinear_sample(const unsigned char* __restrict__ src,
                                                   int src_w, int src_h,
                                                   float src_x, float src_y, int c) {
    if (src_x < 0.0f) src_x = 0.0f;
    if (src_x > (float)(src_w - 1)) src_x = (float)(src_w - 1);
    if (src_y < 0.0f) src_y = 0.0f;
    if (src_y > (float)(src_h - 1)) src_y = (float)(src_h - 1);

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    float v00 = (float)src[(y0 * src_w + x0) * 3 + c];
    float v01 = (float)src[(y0 * src_w + x1) * 3 + c];
    float v10 = (float)src[(y1 * src_w + x0) * 3 + c];
    float v11 = (float)src[(y1 * src_w + x1) * 3 + c];

    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    return v0 + (v1 - v0) * fy;
}

// ---------------------------------------------------------------------
// Kernel 1 (path A, stage 1): bilinear resize, uint8 HWC -> uint8 HWC.
// Scattered taps, no vectorized load (3-channel uint8 pixels do not
// align to a vector width; see README).
// ---------------------------------------------------------------------
__global__ void resize_bilinear_hwc_u8(const unsigned char* __restrict__ src,
                                        unsigned char* __restrict__ dst) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= DST_W || dst_y >= DST_H) return;

    const float scale_x = (float)SRC_W / (float)DST_W;
    const float scale_y = (float)SRC_H / (float)DST_H;
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    int out_base = (dst_y * DST_W + dst_x) * 3;
#pragma unroll
    for (int c = 0; c < 3; c++) {
        float v = bilinear_sample(src, SRC_W, SRC_H, src_x, src_y, c);
        dst[out_base + c] = (unsigned char)(v + 0.5f);
    }
}

// ---------------------------------------------------------------------
// Kernel 2 (path A, stage 2): normalize, uint8 HWC -> float32 HWC.
// The HWC layout is a flat array of DST_H*DST_W*3 bytes, which divides
// evenly by 4 (921600 / 4 = 230400), so every thread does a clean
// vectorized uchar4 load and float4 store, no remainder handling.
// ---------------------------------------------------------------------
__global__ void normalize_hwc_f32(const unsigned char* __restrict__ src,
                                   float* __restrict__ dst) {
    int grp = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_groups = (DST_H * DST_W * 3) / 4;
    if (grp >= total_groups) return;

    int base = grp * 4;
    uchar4 in4 = *reinterpret_cast<const uchar4*>(src + base);
    unsigned char vals[4] = {in4.x, in4.y, in4.z, in4.w};
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int c = (base + i) % 3;
        float norm01 = (float)vals[i] / 255.0f;
        out[i] = (norm01 - d_mean[c]) / d_std[c];
    }
    float4 out4 = make_float4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<float4*>(dst + base) = out4;
}

// ---------------------------------------------------------------------
// Kernel 3 (path A, stage 3): layout transpose, float32 HWC -> CHW.
// DST_W (640) is divisible by 4, so each thread gathers 4 consecutive
// x positions (strided reads from HWC, unavoidable) and issues one
// vectorized float4 store into the contiguous CHW output row.
// ---------------------------------------------------------------------
__global__ void hwc_to_chw_f32(const float* __restrict__ src,
                                float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int xgroups = DST_W / 4;
    const int total = CH * DST_H * xgroups;
    if (idx >= total) return;

    int xg = idx % xgroups;
    int tmp = idx / xgroups;
    int y = tmp % DST_H;
    int c = tmp / DST_H;

    int x0 = xg * 4;
    float v0 = src[(y * DST_W + (x0 + 0)) * 3 + c];
    float v1 = src[(y * DST_W + (x0 + 1)) * 3 + c];
    float v2 = src[(y * DST_W + (x0 + 2)) * 3 + c];
    float v3 = src[(y * DST_W + (x0 + 3)) * 3 + c];

    int out_base = (c * DST_H + y) * DST_W + x0;
    *reinterpret_cast<float4*>(dst + out_base) = make_float4(v0, v1, v2, v3);
}

// ---------------------------------------------------------------------
// Path B: fused resize + normalize + layout, uint8 HWC -> float32 CHW
// in a single kernel and a single pass over global memory (one read of
// the source taps, one write of the CHW output). Same thread mapping
// and vectorized CHW store as hwc_to_chw_f32, but the "read" side does
// bilinear sampling straight out of the uint8 source instead of a
// gather from an intermediate normalized buffer.
// ---------------------------------------------------------------------
__global__ void fused_resize_normalize_chw(const unsigned char* __restrict__ src,
                                            float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int xgroups = DST_W / 4;
    const int total = CH * DST_H * xgroups;
    if (idx >= total) return;

    int xg = idx % xgroups;
    int tmp = idx / xgroups;
    int y = tmp % DST_H;
    int c = tmp / DST_H;

    const float scale_x = (float)SRC_W / (float)DST_W;
    const float scale_y = (float)SRC_H / (float)DST_H;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    int x0 = xg * 4;
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int dst_x = x0 + i;
        float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
        float v = bilinear_sample(src, SRC_W, SRC_H, src_x, src_y, c);
        float norm01 = v / 255.0f;
        out[i] = (norm01 - d_mean[c]) / d_std[c];
    }

    int out_base = (c * DST_H + y) * DST_W + x0;
    *reinterpret_cast<float4*>(dst + out_base) = make_float4(out[0], out[1], out[2], out[3]);
}

// ---------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------

// Deterministic synthetic input pattern, must match parity.py exactly.
// value(y, x, c) = (x*7 + y*13 + c*29) mod 256
static void fill_synthetic_input(std::vector<unsigned char>& img) {
    for (int y = 0; y < SRC_H; y++) {
        for (int x = 0; x < SRC_W; x++) {
            for (int c = 0; c < 3; c++) {
                int v = (x * 7 + y * 13 + c * 29) % 256;
                img[(y * SRC_W + x) * 3 + c] = (unsigned char)v;
            }
        }
    }
}

struct Stats {
    double mean_us;
    double p50_us;
    double p99_us;
};

static Stats compute_stats(std::vector<float>& samples_us) {
    std::sort(samples_us.begin(), samples_us.end());
    size_t n = samples_us.size();
    double sum = 0.0;
    for (float v : samples_us) sum += v;
    Stats s;
    s.mean_us = sum / (double)n;
    s.p50_us = samples_us[(size_t)(0.50 * (n - 1))];
    s.p99_us = samples_us[(size_t)(0.99 * (n - 1))];
    return s;
}

// Templated so the compiler deduces the kernel's own function-pointer
// type (T = void(*)(Args...)) and CUDA's occupancy/attribute template
// wrappers in cuda_runtime.h do the void* forwarding internally. This
// is the pattern NVIDIA's own occupancy-API samples use; passing an
// explicit (const void*) cast at the call site is avoided here since
// it depends on overload resolution details this file cannot verify
// without nvcc on hand.
template <typename KernelFunc>
static void print_kernel_info(const char* name, KernelFunc func, int block_size) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, func));
    int max_active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, func, block_size, 0));
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    double occupancy = (double)(max_active_blocks * block_size) /
                        (double)prop.maxThreadsPerMultiProcessor;
    printf("  %-28s regs/thread=%-4d block=%-4d max_active_blocks/SM=%-3d occupancy=%.2f\n",
           name, attr.numRegs, block_size, max_active_blocks, occupancy);
}

int main(int argc, char** argv) {
    const char* dump_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_path = argv[i + 1];
            i++;
        }
    }

    // Nominal per-frame byte traffic (input + output sizes; overlapping
    // bilinear taps are not counted individually because L1/L2 caching
    // absorbs the reuse, see README).
    const size_t SRC_BYTES = (size_t)SRC_H * SRC_W * 3;                 // uint8
    const size_t DST_U8_BYTES = (size_t)DST_H * DST_W * 3;              // uint8
    const size_t DST_F32_BYTES = (size_t)DST_H * DST_W * 3 * sizeof(float);

    const double bytes_A = (double)SRC_BYTES + 2.0 * (double)DST_U8_BYTES +
                            2.0 * (double)DST_F32_BYTES + (double)DST_F32_BYTES;
    const double bytes_B = (double)SRC_BYTES + (double)DST_F32_BYTES;

    // Host buffers
    std::vector<unsigned char> h_src((size_t)SRC_H * SRC_W * 3);
    fill_synthetic_input(h_src);
    std::vector<float> h_out_b((size_t)DST_H * DST_W * 3);

    // Device buffers
    unsigned char *d_src, *d_resized_u8;
    float *d_normalized_hwc, *d_chw_a, *d_chw_b;
    CUDA_CHECK(cudaMalloc(&d_src, SRC_BYTES));
    CUDA_CHECK(cudaMalloc(&d_resized_u8, DST_U8_BYTES));
    CUDA_CHECK(cudaMalloc(&d_normalized_hwc, DST_F32_BYTES));
    CUDA_CHECK(cudaMalloc(&d_chw_a, DST_F32_BYTES));
    CUDA_CHECK(cudaMalloc(&d_chw_b, DST_F32_BYTES));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), SRC_BYTES, cudaMemcpyHostToDevice));

    // Launch geometry
    dim3 resize_block(16, 16);
    dim3 resize_grid(DST_W / 16, DST_H / 16);
    const int resize_block_size = 16 * 16;

    const int norm_block_size = 256;
    const int norm_total = (DST_H * DST_W * 3) / 4;
    const int norm_grid = (norm_total + norm_block_size - 1) / norm_block_size;

    const int chw_block_size = 256;
    const int chw_total = CH * DST_H * (DST_W / 4);
    const int chw_grid = (chw_total + chw_block_size - 1) / chw_block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        resize_bilinear_hwc_u8<<<resize_grid, resize_block>>>(d_src, d_resized_u8);
        normalize_hwc_f32<<<norm_grid, norm_block_size>>>(d_resized_u8, d_normalized_hwc);
        hwc_to_chw_f32<<<chw_grid, chw_block_size>>>(d_normalized_hwc, d_chw_a);
        fused_resize_normalize_chw<<<chw_grid, chw_block_size>>>(d_src, d_chw_b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed: path A (three kernels, one DRAM round trip per stage)
    std::vector<float> times_a(TIMED_ITERS);
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        resize_bilinear_hwc_u8<<<resize_grid, resize_block>>>(d_src, d_resized_u8);
        normalize_hwc_f32<<<norm_grid, norm_block_size>>>(d_resized_u8, d_normalized_hwc);
        hwc_to_chw_f32<<<chw_grid, chw_block_size>>>(d_normalized_hwc, d_chw_a);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_a[i] = ms * 1000.0f;
    }

    // Timed: path B (one fused kernel, one DRAM round trip)
    std::vector<float> times_b(TIMED_ITERS);
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        fused_resize_normalize_chw<<<chw_grid, chw_block_size>>>(d_src, d_chw_b);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_b[i] = ms * 1000.0f;
    }

    Stats stats_a = compute_stats(times_a);
    Stats stats_b = compute_stats(times_b);
    double gbps_a = bytes_A / (stats_a.mean_us * 1e-6) / 1e9;
    double gbps_b = bytes_B / (stats_b.mean_us * 1e-6) / 1e9;

    printf("Day 7: fused Ampere preprocess kernel benchmark\n");
    printf("Input:  %dx%dx3 uint8 HWC\n", SRC_W, SRC_H);
    printf("Output: 3x%dx%d float32 CHW\n\n", DST_H, DST_W);

    printf("Path A (resize -> normalize -> hwc_to_chw, 3 kernels):\n");
    printf("  mean=%.2f us  p50=%.2f us  p99=%.2f us  bandwidth=%.2f GB/s\n",
           stats_a.mean_us, stats_a.p50_us, stats_a.p99_us, gbps_a);
    printf("Path B (fused_resize_normalize_chw, 1 kernel):\n");
    printf("  mean=%.2f us  p50=%.2f us  p99=%.2f us  bandwidth=%.2f GB/s\n\n",
           stats_b.mean_us, stats_b.p50_us, stats_b.p99_us, gbps_b);
    printf("Speedup (path A mean / path B mean): %.2fx\n\n", stats_a.mean_us / stats_b.mean_us);

    printf("Kernel occupancy (chosen launch block size):\n");
    print_kernel_info("resize_bilinear_hwc_u8", resize_bilinear_hwc_u8, resize_block_size);
    print_kernel_info("normalize_hwc_f32", normalize_hwc_f32, norm_block_size);
    print_kernel_info("hwc_to_chw_f32", hwc_to_chw_f32, chw_block_size);
    print_kernel_info("fused_resize_normalize_chw", fused_resize_normalize_chw, chw_block_size);
    printf("\n");

    if (dump_path != nullptr) {
        CUDA_CHECK(cudaMemcpy(h_out_b.data(), d_chw_b, DST_F32_BYTES, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump_path, "wb");
        if (f == nullptr) {
            fprintf(stderr, "failed to open dump file: %s\n", dump_path);
            exit(1);
        }
        fwrite(h_out_b.data(), sizeof(float), h_out_b.size(), f);
        fclose(f);
        printf("Dumped fused output (float32 CHW, %zu floats) to %s\n\n", h_out_b.size(), dump_path);
    }

    // Single-line machine-readable summary for run.sh to grep.
    printf("RESULT_JSON: {"
           "\"src_w\": %d, \"src_h\": %d, \"dst_w\": %d, \"dst_h\": %d, "
           "\"warmup_iters\": %d, \"timed_iters\": %d, "
           "\"path_a\": {\"mean_us\": %.3f, \"p50_us\": %.3f, \"p99_us\": %.3f, \"gbps\": %.3f}, "
           "\"path_b\": {\"mean_us\": %.3f, \"p50_us\": %.3f, \"p99_us\": %.3f, \"gbps\": %.3f}, "
           "\"speedup\": %.4f}\n",
           SRC_W, SRC_H, DST_W, DST_H, WARMUP_ITERS, TIMED_ITERS,
           stats_a.mean_us, stats_a.p50_us, stats_a.p99_us, gbps_a,
           stats_b.mean_us, stats_b.p50_us, stats_b.p99_us, gbps_b,
           stats_a.mean_us / stats_b.mean_us);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_resized_u8));
    CUDA_CHECK(cudaFree(d_normalized_hwc));
    CUDA_CHECK(cudaFree(d_chw_a));
    CUDA_CHECK(cudaFree(d_chw_b));

    return 0;
}
