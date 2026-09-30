/* Shared by the host code (C) and the kernel (compiled as CUDA C++), so keep it plain. */
#ifndef INK_STATS_H
#define INK_STATS_H

typedef struct {
    unsigned count;          /* pixels darker than the threshold */
    unsigned min_x, min_y;   /* bounding box of those pixels, inclusive */
    unsigned max_x, max_y;
} InkStats;

#endif
