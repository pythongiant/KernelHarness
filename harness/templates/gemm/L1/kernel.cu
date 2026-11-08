// L1: 16x16 shared-memory tile per block, one output element per thread.
#define TILE 16
extern "C" __global__ void mm_kernel(const HarnessParams p) {
    __shared__ @@DTYPE@@ As[TILE][TILE];
    __shared__ @@DTYPE@@ Bs[TILE][TILE];
    long m = @@M@@, k = @@K@@, n = @@N_OUT@@;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    const @@DTYPE@@* A = (const @@DTYPE@@*)p.buf[0];
    const @@DTYPE@@* B = (const @@DTYPE@@*)p.buf[1];
    @@DTYPE@@* C = (@@DTYPE@@*)p.buf[2];
    @@DTYPE@@ acc = 0;
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
    long m = @@M@@, n = @@N_OUT@@;
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((m + TILE - 1) / TILE));
    mm_kernel<<<grid, block>>>(p);
}
