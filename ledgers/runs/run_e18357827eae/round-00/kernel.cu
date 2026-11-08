// L1: 16x16 shared-memory tile per block, one output element per thread.
#define TILE 16
extern "C" __global__ void mm_kernel(const HarnessParams p) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    long m = 512, k = 512, n = 512;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    const float* A = (const float*)p.buf[0];
    const float* B = (const float*)p.buf[1];
    float* C = (float*)p.buf[2];
    float acc = 0;
    for (long t = 0; t < (k + TILE - 1) / TILE; ++t) {
        long ta = t * TILE;
        As[threadIdx.y][threadIdx.x] = (row < m && ta + threadIdx.x < k) ? A[row * k + ta + threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] = (ta + threadIdx.y < k && col < n) ? B[(ta + threadIdx.y) * n + col] : 0;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    dim3 block(TILE, TILE);
    long m = 512, n = 512;
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((m + TILE - 1) / TILE));
    mm_kernel<<<grid, block>>>(p);
}
