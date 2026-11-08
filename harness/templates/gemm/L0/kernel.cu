// L0: scalar matmul, one output element per thread, no tiling.
extern "C" __global__ void mm_kernel(const HarnessParams p) {
    long m = @@M@@, k = @@K@@, n = @@N_OUT@@;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    long row = idx / n, col = idx % n;
    const @@DTYPE@@* A = (const @@DTYPE@@*)p.buf[0];
    const @@DTYPE@@* B = (const @@DTYPE@@*)p.buf[1];
    @@DTYPE@@* C = (@@DTYPE@@*)p.buf[2];
    @@DTYPE@@ acc = 0;
    for (long kk = 0; kk < k; ++kk) acc += A[row * k + kk] * B[kk * n + col];
    C[idx] = acc;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 128;
    long total = @@M@@ * @@N_OUT@@;
    int grid = (int)((total + block - 1) / block);
    mm_kernel<<<grid, block>>>(p);
}
