// L3: vectorized + grid-stride loop + streaming stores (__stcs).
extern "C" __global__ void __launch_bounds__(256, 8)
affine_kernel(const HarnessParams p) {
    long stride = (long)gridDim.x * blockDim.x;
    long n4 = p.dim[0] / 4;
    const int4* x = (const int4*)p.buf[0];
    int4* out = (int4*)p.buf[1];
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n4; t += stride) {
        int4 v = __ldg(&x[t]);
        v.x = @@A@@ * v.x + @@B@@;
        v.y = @@A@@ * v.y + @@B@@;
        v.z = @@A@@ * v.z + @@B@@;
        v.w = @@A@@ * v.w + @@B@@;
        __stcs(&out[t], v);
    }
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 256;
    long n4 = p.dim[0] / 4;
    int grid = (int)((n4 + block - 1) / block);
    if (grid > 8192) grid = 8192;
    affine_kernel<<<grid, block>>>(p);
}
