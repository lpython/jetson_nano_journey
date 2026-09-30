// Compiled to PTX by clang (see Makefile) and loaded at runtime with cuModuleLoad.
// extern "C" keeps the symbol name unmangled so the host can look it up as "threshold_ink".
#include "../src/ink_stats.h"

// One thread per pixel of the NV12 Y (luma) plane, which is already a grayscale image.
// Dark ink on white paper => ink is anything darker than `thr`.
// Writes a mask in MNIST polarity (white digit on black) and accumulates the ink pixel
// count and bounding box: first per block in shared memory, then one global atomic per block.
extern "C" __global__ void threshold_ink(const unsigned char *luma, int pitch, int width, int height,
                                         int thr, unsigned char *mask, int mask_pitch, InkStats *stats)
{
    __shared__ unsigned s_count, s_min_x, s_min_y, s_max_x, s_max_y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) {
        s_count = 0;
        s_min_x = s_min_y = 0xffffffffu;
        s_max_x = s_max_y = 0;
    }
    __syncthreads();

    // x varies fastest across a warp, so a warp reads 32 consecutive bytes of one row (coalesced).
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const bool ink = luma[y * pitch + x] < thr;
        mask[y * mask_pitch + x] = ink ? 255 : 0;
        if (ink) {
            atomicAdd(&s_count, 1u);
            atomicMin(&s_min_x, (unsigned)x);
            atomicMin(&s_min_y, (unsigned)y);
            atomicMax(&s_max_x, (unsigned)x);
            atomicMax(&s_max_y, (unsigned)y);
        }
    }
    __syncthreads();

    if (tid == 0 && s_count) {
        atomicAdd(&stats->count, s_count);
        atomicMin(&stats->min_x, s_min_x);
        atomicMin(&stats->min_y, s_min_y);
        atomicMax(&stats->max_x, s_max_x);
        atomicMax(&stats->max_y, s_max_y);
    }
}
