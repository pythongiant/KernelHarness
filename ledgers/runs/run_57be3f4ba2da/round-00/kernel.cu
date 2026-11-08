// L0: scalar FlashAttention (online softmax), one row per thread.
#include <math.h>

extern "C" __global__ void attn_kernel(const HarnessParams p) {
    long rows = 256, T = 64, D = 64;
    long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* Q = (const float*)p.buf[0];
    const float* K = (const float*)p.buf[1];
    const float* V = (const float*)p.buf[2];
    float* OUT = (float*)p.buf[3];
    float scale = rsqrtf((float)D);

    float m = -INFINITY, l = 0.0f;
    float acc[64];
    for (int d = 0; d < D && d < 64; ++d) acc[d] = 0.0f;
    for (long t = 0; t < T; ++t) {
        float s = 0.0f;
        for (long d = 0; d < D; ++d) s += Q[row * D + d] * K[t * D + d];
        s *= scale;
        float mNew = fmaxf(m, s);
        float corr = expf(m - mNew);
        float p_t = expf(s - mNew);
        l = l * corr + p_t;
        for (long d = 0; d < D; ++d) acc[d] = acc[d] * corr + p_t * V[t * D + d];
        m = mNew;
    }
    for (long d = 0; d < D; ++d) OUT[row * D + d] = acc[d] / l;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 64;
    long rows = 256;
    int grid = (int)((rows + block - 1) / block);
    attn_kernel<<<grid, block>>>(p);
}
