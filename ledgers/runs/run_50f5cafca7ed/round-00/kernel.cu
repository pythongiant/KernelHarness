// Example user kernel: elementwise affine, two elements per thread.
// The signature is fixed by the harness: it declares the buffers in `HarnessParams`
// (see harness/runner/gen.py) and launches whatever the grid geometry requires.
extern "C" __global__ void affine_kernel2(const HarnessParams p) {
    long i = 2 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
    const int* x = (const int*)p.buf[0];
    int* out = (int*)p.buf[1];
    long n = p.dim[0];
    if (i < n) out[i] = 10 * x[i] + 3;
    if (i + 1 < n) out[i + 1] = 10 * x[i + 1] + 3;
}

extern "C" void launch_kernel(const HarnessParams& p) {
    int block = 256;
    long n2 = (p.dim[0] + 1) / 2;
    int grid = (int)((n2 + block - 1) / block);
    affine_kernel2<<<grid, block>>>(p);
}
