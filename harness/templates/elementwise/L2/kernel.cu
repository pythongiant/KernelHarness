// L2: vectorized + register cap for higher occupancy + read-only path (__ldg).
extern "C" __global__ void __launch_bounds__(256, 8)
affine_kernel(const HarnessParams p) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n4 = p.dim[0] / 4;
    if (t >= n4) return;
    const int4* x = (const int4*)p.buf[0];
    int4* out = (int4*)p.buf[1];
    int4 v = __ldg(&x[t]);
    v.x = @@A@@ * v.x + @@B@@;
    v.y = @@A@@ * v.y + @@B@@;
    v.z = @@A@@ * v.z + @@B@@;
    v.w = @@A@@ * v.w + @@B@@;
    out[t] = v;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 256;
    long n4 = p.dim[0] / 4;
    int grid = (int)((n4 + block - 1) / block);
    affine_kernel<<<grid, block>>>(p);
}
