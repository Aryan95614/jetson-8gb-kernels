// shim.cu
// Day 9: fused preprocess kernel wrapped as a shared library with a
// plain extern "C" API, so it can be launched from either a Python host
// (ctypes) or a Rust host (FFI) with identical semantics.
//
// The kernel body below (bilinear_sample + fused_resize_normalize_chw)
// is copied EXACTLY from Day 7's preprocess.cu (same file, same math,
// same compile-time problem size: 1280x720x3 uint8 HWC in, 3x480x640
// float32 CHW out, half-pixel bilinear, ImageNet mean/std normalize).
// See ../day07/preprocess.cu for the original with the three-kernel
// unfused path and the occupancy diagnostics; this file keeps only the
// fused path plus the C API needed to drive it from an external host.
//
// The four C API functions below (fused_init/upload/run/download/free)
// are new for Day 9: fused_init takes runtime src/dst dimensions for a
// general-looking signature, but this build only actually supports the
// one compile-time size baked into the kernel (SRC_W/H, DST_W/H below);
// passing any other size returns an error instead of silently doing the
// wrong thing.
//
// Build: nvcc -O3 -arch=sm_87 -Xcompiler -fPIC -shared -o libfused.so shim.cu
// Target: Jetson Orin Nano (Ampere, sm_87), CUDA 12.6.

#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Fixed problem size -- identical values to Day 7's preprocess.cu.
// ---------------------------------------------------------------------
static const int SRC_W = 1280;
static const int SRC_H = 720;
static const int DST_W = 640;
static const int DST_H = 480;
static const int CH    = 3;

static const size_t SRC_BYTES = (size_t)SRC_H * SRC_W * CH;                    // uint8
static const size_t DST_BYTES = (size_t)DST_H * DST_W * CH * sizeof(float);    // float32

__constant__ float d_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float d_std[3]  = {0.229f, 0.224f, 0.225f};

// ---------------------------------------------------------------------
// Copied EXACTLY from Day 7's preprocess.cu (bilinear_sample).
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
// Copied EXACTLY from Day 7's preprocess.cu (fused_resize_normalize_chw).
// uint8 HWC 1280x720x3 -> float32 CHW 3x480x640, one pass, one kernel.
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
// C API state. Single stream, single pair of persistent device
// buffers -- matches the intended usage: init once, then upload/run in
// a tight per-frame loop, download only for the parity check, free once.
// ---------------------------------------------------------------------
static unsigned char* g_d_src = nullptr;
static float*         g_d_dst = nullptr;
static cudaStream_t   g_stream = nullptr;
static bool           g_initialized = false;

static const int CHW_BLOCK_SIZE = 256;
static const int CHW_TOTAL = CH * DST_H * (DST_W / 4);
static const int CHW_GRID = (CHW_TOTAL + CHW_BLOCK_SIZE - 1) / CHW_BLOCK_SIZE;

extern "C" int fused_init(int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w != SRC_W || src_h != SRC_H || dst_w != DST_W || dst_h != DST_H) {
        fprintf(stderr,
                "shim.cu: fused_init: unsupported size %dx%d -> %dx%d "
                "(this build only supports %dx%d -> %dx%d, baked in at "
                "compile time)\n",
                src_w, src_h, dst_w, dst_h, SRC_W, SRC_H, DST_W, DST_H);
        return -1;
    }
    if (g_initialized) {
        // Idempotent: already set up with the only size this build supports.
        return 0;
    }

    cudaError_t err;
    err = cudaMalloc(&g_d_src, SRC_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_init: cudaMalloc(src) failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&g_d_dst, DST_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_init: cudaMalloc(dst) failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(g_d_src);
        g_d_src = nullptr;
        return -1;
    }
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_init: cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(g_d_src);
        cudaFree(g_d_dst);
        g_d_src = nullptr;
        g_d_dst = nullptr;
        return -1;
    }

    g_initialized = true;
    return 0;
}

extern "C" int fused_upload(const unsigned char* host_hwc) {
    if (!g_initialized) {
        fprintf(stderr, "shim.cu: fused_upload: not initialized\n");
        return -1;
    }
    cudaError_t err = cudaMemcpyAsync(g_d_src, host_hwc, SRC_BYTES,
                                       cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_upload: cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

extern "C" int fused_run(void) {
    if (!g_initialized) {
        fprintf(stderr, "shim.cu: fused_run: not initialized\n");
        return -1;
    }
    fused_resize_normalize_chw<<<CHW_GRID, CHW_BLOCK_SIZE, 0, g_stream>>>(g_d_src, g_d_dst);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_run: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaStreamSynchronize(g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_run: cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

extern "C" int fused_download(float* host_chw) {
    if (!g_initialized) {
        fprintf(stderr, "shim.cu: fused_download: not initialized\n");
        return -1;
    }
    cudaError_t err = cudaMemcpy(host_chw, g_d_dst, DST_BYTES, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "shim.cu: fused_download: cudaMemcpy failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

extern "C" void fused_free(void) {
    if (!g_initialized) return;
    if (g_stream) cudaStreamDestroy(g_stream);
    if (g_d_src) cudaFree(g_d_src);
    if (g_d_dst) cudaFree(g_d_dst);
    g_stream = nullptr;
    g_d_src = nullptr;
    g_d_dst = nullptr;
    g_initialized = false;
}
