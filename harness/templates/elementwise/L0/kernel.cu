// L0: elementwise affine, one element per thread, scalar accesses.
extern "C" __global__ void affine_kernel(const HarnessParams p) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.dim[0]) return;
    const int* x = (const int*)p.buf[0];
    int* out = (int*)p.buf[1];
    out[i] = @@A@@ * x[i] + @@B@@;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 256;
    long n = p.dim[0];
    int grid = (int)((n + block - 1) / block);
    affine_kernel<<<grid, block>>>(p);
}
