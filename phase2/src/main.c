/* Phase 2: a C program that owns the pipeline.
 *
 *   camera -> tee -> queue(4) -> NVENC -> RTP/UDP :5000 to the Mac      (preview, optional)
 *               \-> queue(1, leaky) -> nvvidconv (VIC scale) -> appsink  (inference branch)
 *
 * Each appsink sample is an NVMM buffer. We map it into CUDA with zero copies
 * (dmabuf fd -> EGLImage -> CUDA), threshold the Y plane, and release it.
 * Once a second we print frame rates, drops, frame age and per-stage timings.
 */
#include <getopt.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <nvbuf_utils.h>

#include "gpu.h"

typedef struct {
    const char *host;        /* Mac IP for the preview; NULL = no preview branch */
    int width, height;       /* inference branch size after nvvidconv */
    int threshold;
    long max_frames;         /* 0 = run until Ctrl-C */
    const char *dump_prefix; /* write PGMs of one frame */
    long dump_at;
    int hold_ms;             /* extra time holding each buffer: stand-in for heavier inference */
} Options;

/* Stats for one reporting window (about 1 s). Times in ms. */
typedef struct {
    int frames;
    double age_sum, age_max;   /* capture -> pulled from appsink */
    double map_sum, map_max;   /* dmabuf fd -> CUDA pointer */
    double kern_sum, kern_max; /* kernel + stats readback */
    double held_sum, held_max; /* pulled -> buffer released back to its pool */
    InkStats ink;              /* last frame's */
} StatsWindow;

static volatile sig_atomic_t g_sigint;
static gint g_camera_frames; /* incremented on the camera's streaming thread */

static gint g_encoder_criticals;

/* nvv4l2h264enc on L4T 32.x trips a gst_buffer_resize_range assertion once per encoded frame.
 * The stream is fine; count it instead of printing it. Every other message goes through. */
static void on_gst_critical(const gchar *domain, GLogLevelFlags level, const gchar *message, gpointer data)
{
    if (message && strstr(message, "gst_buffer_resize_range")) {
        g_atomic_int_inc(&g_encoder_criticals);
        return;
    }
    g_log_default_handler(domain, level, message, data);
}

static void on_sigint(int sig)
{
    (void)sig;
    g_sigint++;
}

static GstPadProbeReturn count_camera_frame(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    g_atomic_int_inc(&g_camera_frames);
    return GST_PAD_PROBE_OK;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void add(double *sum, double *max, double v)
{
    *sum += v;
    if (v > *max)
        *max = v;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [options]\n"
            "  -h HOST   send the preview to HOST:5000 (default: no preview)\n"
            "  -s WxH    inference branch size (default 640x360)\n"
            "  -t N      ink threshold on luma 0-255 (default 100)\n"
            "  -n N      stop after N processed frames\n"
            "  -d PREFIX dump PREFIX_y.pgm and PREFIX_mask.pgm of frame 120\n"
            "  -w MS     hold each buffer MS longer (simulated slow inference)\n",
            argv0);
}

static int parse_options(int argc, char **argv, Options *o)
{
    *o = (Options){.width = 640, .height = 360, .threshold = 100, .dump_at = 120};
    int c;
    while ((c = getopt(argc, argv, "h:s:t:n:d:w:")) != -1) {
        switch (c) {
        case 'h': o->host = optarg; break;
        case 's':
            if (sscanf(optarg, "%dx%d", &o->width, &o->height) != 2) {
                usage(argv[0]);
                return -1;
            }
            break;
        case 't': o->threshold = atoi(optarg); break;
        case 'n': o->max_frames = atol(optarg); break;
        case 'd': o->dump_prefix = optarg; break;
        case 'w': o->hold_ms = atoi(optarg); break;
        default: usage(argv[0]); return -1;
        }
    }
    return 0;
}

static GstElement *build_pipeline(const Options *o)
{
    gchar *preview = o->host
        ? g_strdup_printf("t. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! "
                          "nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=4000000 ! "
                          "h264parse ! rtph264pay config-interval=1 pt=96 ! "
                          "udpsink host=%s port=5000 sync=false ",
                          o->host)
        : g_strdup("");
    gchar *desc = g_strdup_printf(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! "
        "tee name=t %s"
        "t. ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        /* bl-output=false: NV12 NVMM defaults to block-linear (tiled), which CUDA sees as
         * CUarrays. Pitch-linear gives plain row-major device pointers the kernel can index. */
        "nvvidconv bl-output=false ! video/x-raw(memory:NVMM),width=%d,height=%d,format=NV12 ! "
        "appsink name=infer max-buffers=1 drop=true sync=false",
        preview, o->width, o->height);

    GError *err = NULL;
    GstElement *pipeline = gst_parse_launch(desc, &err);
    if (err) {
        fprintf(stderr, "pipeline: %s\n%s\n", err->message, desc);
        g_error_free(err);
        gst_clear_object(&pipeline);
    }
    g_free(desc);
    g_free(preview);
    return pipeline;
}

static int process_sample(GstSample *sample, Gpu *gpu, const Options *o, long index,
                          GstClockTime running_now, StatsWindow *w)
{
    double t0 = now_ms();
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstClockTime pts = GST_BUFFER_PTS(buf);
    if (GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(running_now) && running_now > pts)
        add(&w->age_sum, &w->age_max, (double)(running_now - pts) / GST_MSECOND);

    /* For NVMM caps, the mapped "data" is an NvBuffer handle, not pixels. */
    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        fprintf(stderr, "gst_buffer_map failed\n");
        return -1;
    }
    int fd = -1, rc = -1;
    GpuFrame frame;
    InkStats ink;
    if (ExtractFdFromNvBuffer(map.data, &fd) != 0) {
        fprintf(stderr, "ExtractFdFromNvBuffer failed\n");
        goto out;
    }
    if (gpu_map_fd(gpu, fd, &frame) != 0)
        goto out;
    double t1 = now_ms();

    rc = gpu_threshold(gpu, &frame, o->threshold, &ink);
    double t2 = now_ms();
    if (rc == 0 && o->dump_prefix && index == o->dump_at)
        rc = gpu_dump_pgm(gpu, &frame, o->dump_prefix);
    if (o->hold_ms)
        usleep((useconds_t)o->hold_ms * 1000);

    gpu_unmap(gpu, &frame);
    if (rc == 0) {
        add(&w->map_sum, &w->map_max, t1 - t0);
        add(&w->kern_sum, &w->kern_max, t2 - t1);
        w->ink = ink;
        w->frames++;
    }
out:
    gst_buffer_unmap(buf, &map);
    if (rc == 0)
        add(&w->held_sum, &w->held_max, now_ms() - t0);
    return rc;
}

static void print_window(const StatsWindow *w, int camera_frames, double seconds)
{
    int n = w->frames ? w->frames : 1;
    printf("camera %3.0f fps | processed %3.0f fps, dropped %3d | age %5.1f ms (max %5.1f) | "
           "map %4.2f (%4.2f) | kernel %4.2f (%4.2f) | held %5.2f (%5.2f) ms | ",
           camera_frames / seconds, w->frames / seconds, camera_frames - w->frames,
           w->age_sum / n, w->age_max, w->map_sum / n, w->map_max,
           w->kern_sum / n, w->kern_max, w->held_sum / n, w->held_max);
    if (w->frames && w->ink.count)
        printf("ink %u px [%u,%u]-[%u,%u]\n", w->ink.count, w->ink.min_x, w->ink.min_y,
               w->ink.max_x, w->ink.max_y);
    else
        printf("no ink\n");
    fflush(stdout);
}

static void ptx_path_next_to_exe(char *out, size_t size)
{
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = '\0';
    snprintf(out, size, "%s/threshold.ptx", n > 0 ? dirname(exe) : ".");
}

int main(int argc, char **argv)
{
    Options o;
    if (parse_options(argc, argv, &o) != 0)
        return 2;
    gst_init(&argc, &argv);
    g_log_set_handler("GStreamer", G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION,
                      on_gst_critical, NULL);

    char ptx[PATH_MAX];
    ptx_path_next_to_exe(ptx, sizeof ptx);
    Gpu gpu;
    if (gpu_init(&gpu, ptx) != 0) {
        gpu_destroy(&gpu);
        return 1;
    }

    GstElement *pipeline = build_pipeline(&o);
    if (!pipeline) {
        gpu_destroy(&gpu);
        return 1;
    }
    GstElement *tee = gst_bin_get_by_name(GST_BIN(pipeline), "t");
    GstPad *tee_sink = gst_element_get_static_pad(tee, "sink");
    gst_pad_add_probe(tee_sink, GST_PAD_PROBE_TYPE_BUFFER, count_camera_frame, NULL, NULL);
    gst_object_unref(tee_sink);
    gst_object_unref(tee);
    GstAppSink *sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline), "infer"));
    GstBus *bus = gst_element_get_bus(pipeline);

    signal(SIGINT, on_sigint);
    printf("preview=%s infer=%dx%d threshold=%d hold=%dms  (Ctrl-C to stop)\n",
           o.host ? o.host : "off", o.width, o.height, o.threshold, o.hold_ms);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    int rc = 0, eos_sent = 0;
    long processed = 0;
    StatsWindow w = {0};
    double window_start = now_ms();
    int camera_at_start = 0;
    GstClock *clock = NULL;

    for (;;) {
        if (g_sigint >= 2)
            break;
        if ((g_sigint == 1 || (o.max_frames && processed >= o.max_frames)) && !eos_sent) {
            fprintf(stderr, "\nEOS sent (Ctrl-C again to quit)\n");
            gst_element_send_event(pipeline, gst_event_new_eos());
            eos_sent = 1;
        }

        GstMessage *msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *err;
                gchar *dbg;
                gst_message_parse_error(msg, &err, &dbg);
                fprintf(stderr, "ERROR from %s: %s\n%s\n", GST_OBJECT_NAME(msg->src), err->message,
                        dbg ? dbg : "");
                g_error_free(err);
                g_free(dbg);
                rc = 1;
            }
            gst_message_unref(msg);
            break;
        }

        GstSample *sample = gst_app_sink_try_pull_sample(sink, 100 * GST_MSECOND);
        if (sample) {
            if (!clock)
                clock = gst_element_get_clock(pipeline);
            GstClockTime running_now = clock
                ? gst_clock_get_time(clock) - gst_element_get_base_time(pipeline)
                : GST_CLOCK_TIME_NONE;
            int r = process_sample(sample, &gpu, &o, processed, running_now, &w);
            gst_sample_unref(sample);
            if (r != 0) {
                rc = 1;
                break;
            }
            processed++;
        }

        double elapsed = now_ms() - window_start;
        if (elapsed >= 1000.0) {
            int camera_now = g_atomic_int_get(&g_camera_frames);
            print_window(&w, camera_now - camera_at_start, elapsed / 1000.0);
            camera_at_start = camera_now;
            w = (StatsWindow){0};
            window_start = now_ms();
        }
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    if (clock)
        gst_object_unref(clock);
    gst_object_unref(bus);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    gpu_destroy(&gpu);
    printf("processed %ld frames (%d known encoder CRITICALs suppressed)\n", processed,
           g_atomic_int_get(&g_encoder_criticals));
    return rc;
}
