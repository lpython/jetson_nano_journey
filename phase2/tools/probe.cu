// Stripped-down kernels that kbench times next to threshold_ink, to split its cost into parts.
// All share one signature so kbench can pass the same argument array to each.
// Compiled to PTX by clang like kernels/threshold.cu.

// Nothing: the cost of launching and scheduling the grid.
extern "C" __global__ void empty_k(const unsigned char *luma, int pitch, int width, int height,
                                   int thr, unsigned char *mask)
{
}

// Threshold only, one pixel per thread: memory traffic without stats, barriers or atomics.
extern "C" __global__ void thresh1_k(const unsigned char *luma, int pitch, int width, int height,
                                     int thr, unsigned char *mask)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        mask[y * width + x] = luma[y * pitch + x] < thr ? 255 : 0;
}

// Threshold only, four pixels per thread with 4-byte loads and stores.
// Needs width, pitch and both base pointers to be multiples of 4; kbench checks width.
extern "C" __global__ void thresh4_k(const unsigned char *luma, int pitch, int width, int height,
                                     int thr, unsigned char *mask)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const uchar4 v = *(const uchar4 *)(luma + y * pitch + x);
        *(uchar4 *)(mask + y * width + x) =
            make_uchar4(v.x < thr ? 255 : 0, v.y < thr ? 255 : 0, v.z < thr ? 255 : 0, v.w < thr ? 255 : 0);
    }
}
