// L2: shared-memory tiles + double buffering + tensor-core MMA (wmma f16, f32 IO).
#include <mma.h>
using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 16

extern "C" __global__ void __launch_bounds__(256, 2)
mm_kernel(const HarnessParams p) {
    __shared__ __half As[2][BM][BK];   // (m, k)
    __shared__ __half Bs[2][BK][BN];   // (k, n)
    const float* A = (const float*)p.buf[0];
    const float* B = (const float*)p.buf[1];
    float* C = (float*)p.buf[2];
    long m = @@M@@, k = @@K@@, n = @@N_OUT@@;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    int tid = threadIdx.x;

    auto loadStage = [&](int stage, long kb) {
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            long gr = blockRow + r, gk = kb + c;
            As[stage][r][c] = (gr < m && gk < k) ? __float2half(A[gr * k + gk])
                                                : __float2half(0.f);
        }
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN, c = idx % BN;
            long gk = kb + r, gc = blockCol + c;
            Bs[stage][r][c] = (gk < k && gc < n) ? __float2half(B[gk * n + gc])
                                                 : __float2half(0.f);
        }
    };

    loadStage(0, 0);
    __syncthreads();
    int stage = 0;
    for (long kb = 0; kb + BK <= k; kb += BK) {
        int next = stage ^ 1;
        if (kb + 2 * BK <= k) loadStage(next, kb + BK);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, &As[stage][i * 16][0], BK);
                wmma::load_matrix_sync(b, &Bs[stage][0][j * 16], BN);
                wmma::mma_sync(acc[i][j], a, b, acc[i][j]);
            }
        }
        __syncthreads();
        stage = next;
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int r = blockRow + i * 16, c = blockCol + j * 16;
            if (r < m && c < n)
                wmma::store_matrix_sync(&C[r * n + c], acc[i][j], n, wmma::mem_row_major);
        }
}

extern "C" void launch_kernel(const HarnessParams& p) {
    dim3 block(256);
    long m = @@M@@, n = @@N_OUT@@;
    dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
    mm_kernel<<<grid, block>>>(p);
}
