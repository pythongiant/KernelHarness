// L1: K/V staged in shared memory once per block, one query row per thread.
#include <math.h>

#define ROWS_PER_BLOCK 32

extern "C" __global__ void attn_kernel(const HarnessParams p) {
    extern __shared__ float kv[];          // 2 * T * D floats: K then V
    float* sK = kv;
    float* sV = kv + (size_t)@@T@@ * @@D@@;
    long rows = @@N_ROWS@@, T = @@T@@, D = @@D@@;
    long row = (long)blockIdx.x * blockDim.x + threadIdx.x;

    for (long i = threadIdx.x; i < T * D; i += blockDim.x) {
        sK[i] = ((const float*)p.buf[1])[i];
        sV[i] = ((const float*)p.buf[2])[i];
    }
    __syncthreads();
    if (row >= rows) return;

    const float* Q = (const float*)p.buf[0];
    float* OUT = (float*)p.buf[3];
    float scale = rsqrtf((float)D);

    float m = -INFINITY, l = 0.0f;
    float acc[64];
    for (int d = 0; d < 64; ++d) acc[d] = 0.0f;
    for (long t = 0; t < T; ++t) {
        float s = 0.0f;
        for (long d = 0; d < D; ++d) s += Q[row * D + d] * sK[t * D + d];
        s *= scale;
        float mNew = fmaxf(m, s);
        float corr = expf(m - mNew);
        float p_t = expf(s - mNew);
        l = l * corr + p_t;
        for (long d = 0; d < D; ++d) acc[d] = acc[d] * corr + p_t * sV[t * D + d];
        m = mNew;
    }
    for (long d = 0; d < D; ++d) OUT[row * D + d] = acc[d] / l;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = ROWS_PER_BLOCK;
    long rows = @@N_ROWS@@;
    int grid = (int)((rows + block - 1) / block);
    size_t smem = 2 * (size_t)@@T@@ * @@D@@ * sizeof(float);
    attn_kernel<<<grid, block, smem>>>(p);
}
