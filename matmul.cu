#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.f;

    for(int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if(row < N && t*TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.f;

        if(col < N && t*TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.f;
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if(row < N && col < N)
        C[row*N + col] = sum;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N;
    std::cin >> N;

    int sz = N*N * sizeof(float);

    float *h_A = (float*) malloc(sz);
    float *h_B = (float*) malloc(sz);
    float *h_C = (float*) malloc(sz);

    for(int i=0; i<N*N; i++)
        std::cin >> h_A[i];
    for(int i=0; i<N*N; i++)
        std::cin >> h_B[i];

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sz);
    cudaMalloc((void **) &d_B, sz);
    cudaMalloc((void **) &d_C, sz);

    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + 31)/32, (N + 31)/32);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(2);
    for(int i=0; i<N*N; i++)
        std::cout << h_C[i] << ((i+1) % N == 0 ? '\n' : ' ');

    return 0;
}

