#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_DIM 32     //for matrix_transpose_kernel, matrix_multiplication_kernel, matrix_muldiv_kernel
#define BLOCK_ROWS 8    // for matrix_transpose_kernel
#define BLOCK_DIM_Y 256 // for softmax_kernel

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for(int d = 0; d < TILE_DIM; d += BLOCK_ROWS) {
        if(x < cols && y+d < rows) {
            tile[threadIdx.y+d][threadIdx.x] = input[(y+d)*cols + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int d = 0; d < TILE_DIM; d += BLOCK_ROWS) {
        if(x < rows && y+d < cols) {
            output[(y+d)*rows + x] = tile[threadIdx.x][threadIdx.y+d];
        }
    }
}

__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    int warp_id = ty/32;
    __shared__ float reduction[BLOCK_DIM_Y/32];

    if(row >= M) return;

    float maxval = -INFINITY;

    int vec_end = (N / 4) * 4;
    for(int i = ty; i < N/4; i += BLOCK_DIM_Y) {
        float4 val = reinterpret_cast<const float4*>(&input[row*N + i*4])[0];
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
    }

    for(int i = vec_end + ty; i < N; i += BLOCK_DIM_Y) {
        maxval = fmaxf(maxval, input[row*N + i]);
    }

    for(int mask = 16; mask > 0; mask /= 2) {
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if(ty % 32 == 0) {
        reduction[warp_id] = maxval;
    }
    __syncthreads();

    if(warp_id == 0) {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : -INFINITY;
        for(int mask = 16; mask > 0; mask /= 2) {
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if(ty == 0) {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];

    float divisor = 0.f;

    for(int i = ty; i < N/4; i += BLOCK_DIM_Y) {
        float4 val = reinterpret_cast<const float4*>(&input[row*N + i*4])[0];
        divisor += __expf(val.x - maxval);
        divisor += __expf(val.y - maxval);
        divisor += __expf(val.z - maxval);
        divisor += __expf(val.w - maxval);
    }

    for(int i = vec_end + ty; i < N; i += BLOCK_DIM_Y) {
        divisor += __expf(input[row*N + i] - maxval);
    }

    for(int mask = 16; mask > 0; mask /= 2) {
        divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if(ty % 32 == 0) {
        reduction[warp_id] = divisor;
    }
    __syncthreads();

    if(warp_id == 0) {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for(int mask = 16; mask > 0; mask /= 2) {
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if(ty == 0) {
        reduction[0] = divisor;
    }
    __syncthreads();
    divisor = reduction[0];

    for(int i = ty; i < N/4; i += BLOCK_DIM_Y) {
        float4 val = reinterpret_cast<const float4*>(&input[row*N + i*4])[0];
        val.x = __expf(val.x - maxval) / divisor;
        val.y = __expf(val.y - maxval) / divisor;
        val.z = __expf(val.z - maxval) / divisor;
        val.w = __expf(val.w - maxval) / divisor;
        reinterpret_cast<float4*>(&output[row*N + i*4])[0] = val;
    }

    for(int i = vec_end + ty; i < N; i += BLOCK_DIM_Y) {
        output[row*N + i] = __expf(input[row*N + i] - maxval) / divisor;
    }
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.f;

    for(int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; t++) {
        if(row < M && t*TILE_DIM + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_DIM + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.f;

        if(col < K && t*TILE_DIM + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_DIM + threadIdx.y)*K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.f;
        __syncthreads();

        for(int k = 0; k < TILE_DIM; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if(row < M && col < K)
        C[row*K + col] = sum;
}

__global__ void matrix_muldiv_kernel(const float* A, const float* B, float* C, int M, int N, int K, float d) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.f;

    for(int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; t++) {
        if(row < M && t*TILE_DIM + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_DIM + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.f;

        if(col < K && t*TILE_DIM + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_DIM + threadIdx.y)*K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.f;
        __syncthreads();

        for(int k = 0; k < TILE_DIM; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if(row < M && col < K)
        C[row*K + col] = sum / sqrtf(d);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int M, N;
    std::cin >> M >> N;

    int sz = M*N * sizeof(float);

    float *h_Q = (float*) malloc(sz);
    float *h_K = (float*) malloc(sz);
    float *h_V = (float*) malloc(sz);
    float *out = (float*) malloc(sz);

    for(int i=0; i<M*N; i++)
        std::cin >> h_Q[i];
    for(int i=0; i<M*N; i++)
        std::cin >> h_K[i];
    for(int i=0; i<M*N; i++)
        std::cin >> h_V[i];

    float *d_Q, *d_K, *d_V, *ans;
    cudaMalloc((void **) &d_Q, sz);
    cudaMalloc((void **) &d_K, sz);
    cudaMalloc((void **) &d_V, sz);
    cudaMalloc((void **) &ans, sz);

    cudaMemcpy(d_Q, h_Q, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sz, cudaMemcpyHostToDevice);

    float* K_t;
    cudaMalloc(&K_t, sz);
    dim3 transpose_blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 transpose_gridDim((N + TILE_DIM - 1) / TILE_DIM,
                           (M + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose_kernel<<<transpose_gridDim, transpose_blockDim>>>(d_K, K_t, M, N);

    float* QK;
    cudaMalloc(&QK, M*M * sizeof(float));
    dim3 matmul_blockDim(32, 32);
    dim3 matmul_gridDim((M + matmul_blockDim.x - 1) / matmul_blockDim.x,
                        (M + matmul_blockDim.y - 1) / matmul_blockDim.y);
    matrix_muldiv_kernel<<<matmul_gridDim, matmul_blockDim>>>(d_Q, K_t, QK, M, N, M, N);

    float* QK_softmax;
    cudaMalloc(&QK_softmax, M*M * sizeof(float));
    dim3 softmax_blockDim(1, BLOCK_DIM_Y, 1);
    softmax_kernel<<<M, softmax_blockDim>>>(QK, QK_softmax, M, M);

    matmul_gridDim = dim3((N + matmul_blockDim.x - 1) / matmul_blockDim.x,
                          (M + matmul_blockDim.y - 1) / matmul_blockDim.y);
    matrix_multiplication_kernel<<<matmul_gridDim, matmul_blockDim>>>(QK_softmax, d_V, ans, M, M, N);

    cudaMemcpy(out, ans, sz, cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(4);
    for(int i=0; i<M*N; i++)
        std::cout << out[i] << ((i+1) % N == 0 ? '\n' : ' ');

    return 0;
}

