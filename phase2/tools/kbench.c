/* kbench: measure threshold_ink on the Nano's GPU without the camera.
 *
 *   1. device     limits reported by the driver
 *   2. kernel     registers / shared / local memory of the compiled threshold_ink
 *   3. sweep      threshold_ink over many block shapes, on a frame with ink and a blank one
 *   4. probes     empty / threshold-only kernels on the same grid, to split the cost into parts
 *   5. host path  reset + launch + readback timed from the CPU, the way gpu_threshold() does it
 *
 * Loads build/threshold.ptx and build/probe.ptx from next to the executable.
 * Design and interpretation: tools/kbench.md.
 */
#include <getopt.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#include "../src/ink_stats.h"

#define WARMUP 10
#define PEAK_GBPS 25.6 /* Nano LPDDR4: 64-bit x 3200 MT/s. The driver's MEMORY_CLOCK_RATE is bogus on Tegra. */
#define DEFAULT_SHAPES "32x8,16x16,64x4,32x4,128x1,256x1,32x16,32x32,8x8,8x32,32x1,1x256"

#define CHECK(call)                                                                              \
    do {                                                                                         \
        CUresult r_ = (call);                                                                    \
        if (r_ != CUDA_SUCCESS) {                                                                \
            const char *n_ = NULL;                                                               \
            cuGetErrorName(r_, &n_);                                                             \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, n_ ? n_ : "?");      \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

typedef struct {
    int width, height, pitch, reps, thr;
    const char *shapes;
    unsigned sched_flag;
    const char *sched_name;
} Options;

typedef struct {
    float min_us, med_us;
} Timing;

typedef struct {
    CUfunction fn;
    unsigned gx, gy, bx, by;
    void **args;
} Launch;

/* The synthetic frame: light background with one dark rectangle, so the right answer is known. */
typedef struct {
    unsigned x0, y0, x1, y1; /* ink box, half-open */
} InkBox;

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [options]\n"
            "  -s WxH      frame size (default 640x360)\n"
            "  -p PITCH    row pitch in bytes (default: width rounded up to 256, like nvvidconv)\n"
            "  -b LIST     block shapes, e.g. 32x8,16x16 (default: a 12-shape sweep)\n"
            "  -r N        timed repetitions per measurement (default 200, plus %d warm-up)\n"
            "  -S MODE     context scheduling: auto|spin|yield|blocking (default auto)\n",
            argv0, WARMUP);
}

static int parse_options(int argc, char **argv, Options *o)
{
    *o = (Options){.width = 640, .height = 360, .reps = 200, .thr = 100, .shapes = DEFAULT_SHAPES,
                   .sched_flag = CU_CTX_SCHED_AUTO, .sched_name = "auto"};
    int c;
    while ((c = getopt(argc, argv, "s:p:b:r:S:")) != -1) {
        switch (c) {
        case 's':
            if (sscanf(optarg, "%dx%d", &o->width, &o->height) != 2)
                return -1;
            break;
        case 'p': o->pitch = atoi(optarg); break;
        case 'b': o->shapes = optarg; break;
        case 'r': o->reps = atoi(optarg); break;
        case 'S':
            o->sched_name = optarg;
            if (!strcmp(optarg, "auto")) o->sched_flag = CU_CTX_SCHED_AUTO;
            else if (!strcmp(optarg, "spin")) o->sched_flag = CU_CTX_SCHED_SPIN;
            else if (!strcmp(optarg, "yield")) o->sched_flag = CU_CTX_SCHED_YIELD;
            else if (!strcmp(optarg, "blocking")) o->sched_flag = CU_CTX_SCHED_BLOCKING_SYNC;
            else return -1;
            break;
        default: return -1;
        }
    }
    if (o->pitch == 0)
        o->pitch = (o->width + 255) / 256 * 256;
    if (o->width <= 0 || o->height <= 0 || o->pitch < o->width || o->reps <= 0)
        return -1;
    return 0;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static Timing summarize(float *us, int n)
{
    qsort(us, (size_t)n, sizeof *us, cmp_float);
    return (Timing){us[0], us[n / 2]};
}

static void launch(const Launch *l)
{
    CHECK(cuLaunchKernel(l->fn, l->gx, l->gy, 1, l->bx, l->by, 1, 0, NULL, l->args, NULL));
}

/* GPU time of the kernel alone: events recorded right before and after the launch.
 * If reset_dst is set, the stats are reset before each rep, outside the timed region. */
static Timing time_gpu(const Launch *l, int reps, CUdeviceptr reset_dst, const InkStats *reset_val)
{
    CUevent e0, e1;
    CHECK(cuEventCreate(&e0, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&e1, CU_EVENT_DEFAULT));
    float *us = malloc(sizeof *us * (size_t)reps);
    for (int i = -WARMUP; i < reps; i++) {
        if (reset_dst)
            CHECK(cuMemcpyHtoD(reset_dst, reset_val, sizeof *reset_val));
        CHECK(cuEventRecord(e0, NULL));
        launch(l);
        CHECK(cuEventRecord(e1, NULL));
        CHECK(cuEventSynchronize(e1));
        float ms;
        CHECK(cuEventElapsedTime(&ms, e0, e1));
        if (i >= 0)
            us[i] = ms * 1000.0f;
    }
    Timing t = summarize(us, reps);
    free(us);
    cuEventDestroy(e0);
    cuEventDestroy(e1);
    return t;
}

/* CPU wall time of what gpu_threshold() does per frame: reset stats, launch, read stats back. */
static Timing time_host_path(const Launch *l, int reps, CUdeviceptr d_stats, const InkStats *init)
{
    float *us = malloc(sizeof *us * (size_t)reps);
    InkStats out;
    for (int i = -WARMUP; i < reps; i++) {
        double t0 = now_us();
        CHECK(cuMemcpyHtoD(d_stats, init, sizeof *init));
        launch(l);
        CHECK(cuMemcpyDtoH(&out, d_stats, sizeof out));
        if (i >= 0)
            us[i] = (float)(now_us() - t0);
    }
    Timing t = summarize(us, reps);
    free(us);
    return t;
}

static void print_device(CUdevice dev)
{
    static const struct {
        CUdevice_attribute attr;
        const char *name;
    } attrs[] = {
        {CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs"},
        {CU_DEVICE_ATTRIBUTE_WARP_SIZE, "warp size"},
        {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "max threads/block"},
        {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "max resident threads/SM"},
        {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers/SM"},
        {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, "registers/block"},
        {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared bytes/SM"},
        {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared bytes/block"},
        {CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "L2 bytes"},
        {CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "GPU clock kHz"},
        {CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, "memory bus bits"},
        {CU_DEVICE_ATTRIBUTE_INTEGRATED, "integrated (shares DRAM)"},
    };
    char name[128];
    int major, minor;
    CHECK(cuDeviceGetName(name, sizeof name, dev));
    CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    printf("== device: %s, compute capability %d.%d\n", name, major, minor);
    for (size_t i = 0; i < sizeof attrs / sizeof attrs[0]; i++) {
        int v;
        CHECK(cuDeviceGetAttribute(&v, attrs[i].attr, dev));
        printf("  %-26s %d\n", attrs[i].name, v);
    }
    printf("  %-26s %d (compute capability 5.x table; not queryable in CUDA 10.2)\n",
           "max resident blocks/SM", 32);
}

static void print_kernel(CUfunction fn, const char *name)
{
    int regs, shared, local;
    CHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    CHECK(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
    CHECK(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    printf("\n== kernel %s: %d registers/thread, %d B static shared/block, %d B local/thread%s\n",
           name, regs, shared, local, local ? "  <- register spills" : "");
}

static CUdeviceptr upload_frame(const Options *o, const InkBox *box)
{
    size_t size = (size_t)o->pitch * o->height;
    unsigned char *host = malloc(size);
    for (int y = 0; y < o->height; y++)
        for (int x = 0; x < o->pitch; x++) {
            int inside = box && (unsigned)x >= box->x0 && (unsigned)x < box->x1 &&
                         (unsigned)y >= box->y0 && (unsigned)y < box->y1;
            host[(size_t)y * o->pitch + x] = inside ? 20 : 200;
        }
    CUdeviceptr d;
    CHECK(cuMemAlloc(&d, size));
    CHECK(cuMemcpyHtoD(d, host, size));
    free(host);
    return d;
}

/* Compare the kernel's stats with what the synthetic frame must produce. */
static const char *check(CUdeviceptr d_stats, const InkBox *box)
{
    InkStats s;
    CHECK(cuMemcpyDtoH(&s, d_stats, sizeof s));
    if (!box)
        return s.count == 0 ? "ok" : "FAIL";
    unsigned expect = (box->x1 - box->x0) * (box->y1 - box->y0);
    return s.count == expect && s.min_x == box->x0 && s.min_y == box->y0 && s.max_x == box->x1 - 1 &&
                   s.max_y == box->y1 - 1
               ? "ok"
               : "FAIL";
}

static void ptx_next_to_exe(char *out, size_t size, const char *file)
{
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = '\0';
    snprintf(out, size, "%s/%s", n > 0 ? dirname(exe) : ".", file);
}

int main(int argc, char **argv)
{
    Options o;
    if (parse_options(argc, argv, &o) != 0) {
        usage(argv[0]);
        return 2;
    }

    CUdevice dev;
    CUcontext ctx;
    CUmodule thr_mod, probe_mod;
    CUfunction threshold, empty_k, thresh1_k, thresh4_k;
    char path[PATH_MAX];
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, o.sched_flag, dev));
    ptx_next_to_exe(path, sizeof path, "threshold.ptx");
    CHECK(cuModuleLoad(&thr_mod, path));
    ptx_next_to_exe(path, sizeof path, "probe.ptx");
    CHECK(cuModuleLoad(&probe_mod, path));
    CHECK(cuModuleGetFunction(&threshold, thr_mod, "threshold_ink"));
    CHECK(cuModuleGetFunction(&empty_k, probe_mod, "empty_k"));
    CHECK(cuModuleGetFunction(&thresh1_k, probe_mod, "thresh1_k"));
    CHECK(cuModuleGetFunction(&thresh4_k, probe_mod, "thresh4_k"));

    print_device(dev);
    print_kernel(threshold, "threshold_ink");

    /* Ink box: the middle third horizontally, middle half vertically (about 17% of the frame). */
    InkBox box = {(unsigned)o.width / 3, (unsigned)o.height / 4, (unsigned)o.width * 2 / 3,
                  (unsigned)o.height * 3 / 4};
    unsigned ink_px = (box.x1 - box.x0) * (box.y1 - box.y0);
    CUdeviceptr d_ink = upload_frame(&o, &box), d_blank = upload_frame(&o, NULL);
    CUdeviceptr d_mask, d_stats;
    CHECK(cuMemAlloc(&d_mask, (size_t)o.width * o.height));
    CHECK(cuMemAlloc(&d_stats, sizeof(InkStats)));
    const InkStats init = {0, UINT_MAX, UINT_MAX, 0, 0};

    double bytes = 2.0 * o.width * o.height; /* read luma + write mask */
    printf("\n== frame %dx%d, pitch %d, ink box %ux%u = %u px (%.1f%%)\n", o.width, o.height, o.pitch,
           box.x1 - box.x0, box.y1 - box.y0, ink_px, 100.0 * ink_px / (o.width * o.height));
    printf("   floor: %.0f KB moved -> %.1f us at %.1f GB/s peak\n", bytes / 1024,
           bytes / (PEAK_GBPS * 1e3), PEAK_GBPS);

    /* Kernel arguments. `luma` is swapped between the ink and blank frames; args points at it. */
    CUdeviceptr luma = d_ink;
    int pitch = o.pitch, width = o.width, height = o.height, thr = o.thr, mask_pitch = o.width;
    void *thr_args[] = {&luma, &pitch, &width, &height, &thr, &d_mask, &mask_pitch, &d_stats};
    void *probe_args[] = {&luma, &pitch, &width, &height, &thr, &d_mask};

    printf("\n== block shape sweep: threshold_ink, median (min) of %d reps, GPU time only\n", o.reps);
    printf("%-7s %5s %9s %6s %6s %6s %5s %16s %16s %6s  %s\n", "block", "thr", "grid", "blocks",
           "idle%", "blk/SM", "occ", "ink us", "blank us", "GB/s", "check");
    char *shapes = strdup(o.shapes), *save = NULL;
    for (char *tok = strtok_r(shapes, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        unsigned bx, by;
        if (sscanf(tok, "%ux%u", &bx, &by) != 2 || bx * by == 0 || bx * by > 1024) {
            printf("%-7s skipped (need 1..1024 threads)\n", tok);
            continue;
        }
        Launch l = {threshold, (o.width + bx - 1) / bx, (o.height + by - 1) / by, bx, by, thr_args};
        double launched = (double)l.gx * bx * l.gy * by;
        int per_sm = 0;
        CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, threshold, (int)(bx * by), 0));

        luma = d_ink;
        Timing ink = time_gpu(&l, o.reps, d_stats, &init);
        const char *ok_ink = check(d_stats, &box);
        luma = d_blank;
        Timing blank = time_gpu(&l, o.reps, d_stats, &init);
        const char *ok_blank = check(d_stats, NULL);

        char grid[32];
        snprintf(grid, sizeof grid, "%ux%u", l.gx, l.gy);
        printf("%-7s %5u %9s %6u %5.1f%% %6d %4.0f%% %7.1f (%6.1f) %7.1f (%6.1f) %6.2f  %s/%s\n", tok,
               bx * by, grid, l.gx * l.gy, 100.0 * (launched - (double)o.width * o.height) / launched,
               per_sm, 100.0 * per_sm * bx * by / 2048, ink.med_us, ink.min_us, blank.med_us,
               blank.min_us, bytes / (ink.med_us * 1e3), ok_ink, ok_blank);
    }
    free(shapes);

    printf("\n== probes: 32x8 blocks, blank frame, median (min) GPU time\n");
    luma = d_blank;
    Launch l_empty = {empty_k, (o.width + 31) / 32, (o.height + 7) / 8, 32, 8, probe_args};
    Launch l_t1 = {thresh1_k, l_empty.gx, l_empty.gy, 32, 8, probe_args};
    Timing t_empty = time_gpu(&l_empty, o.reps, 0, NULL);
    Timing t_t1 = time_gpu(&l_t1, o.reps, 0, NULL);
    printf("  %-34s %4u blocks %7.1f (%6.1f) us\n", "empty_k (launch + scheduling)", l_empty.gx * l_empty.gy,
           t_empty.med_us, t_empty.min_us);
    printf("  %-34s %4u blocks %7.1f (%6.1f) us  %5.2f GB/s\n", "thresh1_k (1 px/thread, no stats)",
           l_t1.gx * l_t1.gy, t_t1.med_us, t_t1.min_us, bytes / (t_t1.med_us * 1e3));
    if (o.width % 4 == 0 && o.pitch % 4 == 0) {
        Launch l_t4 = {thresh4_k, (o.width / 4 + 31) / 32, (o.height + 7) / 8, 32, 8, probe_args};
        Timing t_t4 = time_gpu(&l_t4, o.reps, 0, NULL);
        printf("  %-34s %4u blocks %7.1f (%6.1f) us  %5.2f GB/s\n", "thresh4_k (4 px/thread, uchar4)",
               l_t4.gx * l_t4.gy, t_t4.med_us, t_t4.min_us, bytes / (t_t4.med_us * 1e3));
    } else {
        printf("  thresh4_k skipped: width and pitch must be multiples of 4\n");
    }

    printf("\n== host path: reset + launch + readback, as gpu_threshold() does (32x8, ink frame, sched=%s)\n",
           o.sched_name);
    luma = d_ink;
    Launch l_host = {threshold, (o.width + 31) / 32, (o.height + 7) / 8, 32, 8, thr_args};
    Timing gpu = time_gpu(&l_host, o.reps, d_stats, &init);
    Timing host = time_host_path(&l_host, o.reps, d_stats, &init);
    printf("  CPU wall %7.1f (%6.1f) us = GPU kernel %7.1f us + %7.1f us of copies, launch and waiting\n",
           host.med_us, host.min_us, gpu.med_us, host.med_us - gpu.med_us);

    cuMemFree(d_ink);
    cuMemFree(d_blank);
    cuMemFree(d_mask);
    cuMemFree(d_stats);
    cuModuleUnload(probe_mod);
    cuModuleUnload(thr_mod);
    cuCtxDestroy(ctx);
    return 0;
}
