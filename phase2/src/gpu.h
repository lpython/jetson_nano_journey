/* EGL + CUDA driver API side of Phase 2: map an NVMM dmabuf into CUDA with zero copies,
 * run the threshold kernel on it, and release it. All calls must come from one thread,
 * the one that called gpu_init (the CUDA context is current there). */
#ifndef GPU_H
#define GPU_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>

#include "ink_stats.h"

typedef struct {
    EGLDisplay egl;
    CUcontext ctx;
    CUmodule module;
    CUfunction threshold;
    CUdeviceptr d_stats;           /* one InkStats */
    CUdeviceptr d_mask;            /* mask_w * mask_h bytes, allocated on first frame */
    unsigned mask_w, mask_h;
} Gpu;

/* One camera frame mapped into CUDA. Holding it holds the NVMM buffer: release fast. */
typedef struct {
    EGLImageKHR image;
    CUgraphicsResource resource;
    CUeglFrame frame;              /* frame.frame.pPitch[0] = Y plane, frame.pitch = its row stride */
} GpuFrame;

int gpu_init(Gpu *g, const char *ptx_path);
void gpu_destroy(Gpu *g);

int gpu_map_fd(Gpu *g, int dmabuf_fd, GpuFrame *f);
void gpu_unmap(Gpu *g, GpuFrame *f);

int gpu_threshold(Gpu *g, const GpuFrame *f, int thr, InkStats *out);
/* Writes <prefix>_y.pgm (the luma the GPU saw) and <prefix>_mask.pgm (the last threshold mask). */
int gpu_dump_pgm(Gpu *g, const GpuFrame *f, const char *prefix);

#endif
