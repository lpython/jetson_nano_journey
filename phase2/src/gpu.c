#include "gpu.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nvbuf_utils.h>

static int cu_ok(CUresult r, const char *expr, const char *file, int line)
{
    if (r == CUDA_SUCCESS)
        return 1;
    const char *name = NULL;
    cuGetErrorName(r, &name);
    fprintf(stderr, "%s:%d: %s -> %s\n", file, line, expr, name ? name : "?");
    return 0;
}
#define CU_OK(call) cu_ok((call), #call, __FILE__, __LINE__)

int gpu_init(Gpu *g, const char *ptx_path)
{
    memset(g, 0, sizeof *g);

    /* NvEGLImageFromFd needs an initialized EGL display; no window or X session is required. */
    g->egl = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g->egl == EGL_NO_DISPLAY || !eglInitialize(g->egl, NULL, NULL)) {
        fprintf(stderr, "EGL init failed (0x%x)\n", eglGetError());
        return -1;
    }

    CUdevice dev;
    if (!CU_OK(cuInit(0)) || !CU_OK(cuDeviceGet(&dev, 0)) || !CU_OK(cuCtxCreate(&g->ctx, 0, dev)))
        return -1;
    /* The driver JIT-compiles the PTX for sm_53 here (and caches it in ~/.nv/ComputeCache). */
    if (!CU_OK(cuModuleLoad(&g->module, ptx_path)) ||
        !CU_OK(cuModuleGetFunction(&g->threshold, g->module, "threshold_ink")) ||
        !CU_OK(cuMemAlloc(&g->d_stats, sizeof(InkStats))))
        return -1;
    return 0;
}

void gpu_destroy(Gpu *g)
{
    if (g->d_mask)
        cuMemFree(g->d_mask);
    if (g->d_stats)
        cuMemFree(g->d_stats);
    if (g->module)
        cuModuleUnload(g->module);
    if (g->ctx)
        cuCtxDestroy(g->ctx);
    if (g->egl != EGL_NO_DISPLAY)
        eglTerminate(g->egl);
    memset(g, 0, sizeof *g);
}

int gpu_map_fd(Gpu *g, int dmabuf_fd, GpuFrame *f)
{
    memset(f, 0, sizeof *f);

    /* dmabuf fd -> EGLImage -> CUDA graphics resource -> device pointers. No pixel is copied. */
    f->image = NvEGLImageFromFd(g->egl, dmabuf_fd);
    if (!f->image) {
        fprintf(stderr, "NvEGLImageFromFd(fd=%d) failed\n", dmabuf_fd);
        return -1;
    }
    if (!CU_OK(cuGraphicsEGLRegisterImage(&f->resource, f->image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)) ||
        !CU_OK(cuGraphicsResourceGetMappedEglFrame(&f->frame, f->resource, 0, 0)) ||
        !CU_OK(cuCtxSynchronize()))
        goto fail;

    /* Needs nvvidconv bl-output=false. Block-linear arrives as CUarrays (frameType ARRAY) instead. */
    if (f->frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
        fprintf(stderr, "expected a pitch-linear frame, got frameType=%d\n", (int)f->frame.frameType);
        goto fail;
    }
    return 0;

fail:
    gpu_unmap(g, f);
    return -1;
}

void gpu_unmap(Gpu *g, GpuFrame *f)
{
    if (f->resource)
        cuGraphicsUnregisterResource(f->resource);
    if (f->image)
        NvDestroyEGLImage(g->egl, f->image);
    memset(f, 0, sizeof *f);
}

int gpu_threshold(Gpu *g, const GpuFrame *f, int thr, InkStats *out)
{
    int width = (int)f->frame.width, height = (int)f->frame.height, pitch = (int)f->frame.pitch;

    if (g->mask_w != f->frame.width || g->mask_h != f->frame.height) {
        if (g->d_mask)
            cuMemFree(g->d_mask);
        g->d_mask = 0;
        if (!CU_OK(cuMemAlloc(&g->d_mask, (size_t)width * height)))
            return -1;
        g->mask_w = f->frame.width;
        g->mask_h = f->frame.height;
    }

    InkStats init = {0, UINT_MAX, UINT_MAX, 0, 0};
    if (!CU_OK(cuMemcpyHtoD(g->d_stats, &init, sizeof init)))
        return -1;

    CUdeviceptr luma = (CUdeviceptr)(uintptr_t)f->frame.frame.pPitch[0];
    int mask_pitch = width;
    void *args[] = {&luma, &pitch, &width, &height, &thr, &g->d_mask, &mask_pitch, &g->d_stats};
    const unsigned bx = 32, by = 8;
    if (!CU_OK(cuLaunchKernel(g->threshold, (width + bx - 1) / bx, (height + by - 1) / by, 1,
                              bx, by, 1, 0, NULL, args, NULL)))
        return -1;
    /* Synchronous copy: also waits for the kernel to finish. */
    return CU_OK(cuMemcpyDtoH(out, g->d_stats, sizeof *out)) ? 0 : -1;
}

static int write_pgm(const char *path, const unsigned char *pixels, unsigned w, unsigned h)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        return -1;
    }
    fprintf(fp, "P5\n%u %u\n255\n", w, h);
    size_t n = fwrite(pixels, 1, (size_t)w * h, fp);
    fclose(fp);
    return n == (size_t)w * h ? 0 : -1;
}

int gpu_dump_pgm(Gpu *g, const GpuFrame *f, const char *prefix)
{
    unsigned w = f->frame.width, h = f->frame.height;
    unsigned char *host = malloc((size_t)w * h);
    char path[512];
    int rc = -1;
    if (!host)
        return -1;

    CUDA_MEMCPY2D copy = {0};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = (CUdeviceptr)(uintptr_t)f->frame.frame.pPitch[0];
    copy.srcPitch = f->frame.pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = host;
    copy.dstPitch = w;
    copy.WidthInBytes = w;
    copy.Height = h;
    snprintf(path, sizeof path, "%s_y.pgm", prefix);
    if (!CU_OK(cuMemcpy2D(&copy)) || write_pgm(path, host, w, h) != 0)
        goto out;

    snprintf(path, sizeof path, "%s_mask.pgm", prefix);
    if (!CU_OK(cuMemcpyDtoH(host, g->d_mask, (size_t)w * h)) || write_pgm(path, host, w, h) != 0)
        goto out;
    fprintf(stderr, "wrote %s_y.pgm and %s_mask.pgm (%ux%u)\n", prefix, prefix, w, h);
    rc = 0;
out:
    free(host);
    return rc;
}
