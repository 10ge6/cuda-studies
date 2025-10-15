#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

__device__ void insert(float *arr, int idx, float val) {
    if(val <= arr[idx-1]) {
        return;
    }

    for(int i = idx-2; i >= 0; i--) {
        if(val > arr[i]) {
            arr[i+1] = arr[i];
        } else {
            arr[i+1] = val;
            return;
        }
    }
    arr[0] = val;
}

__global__ void rtopk_kernel(const float *matrix, float *output, int M, int N, int K) {
    int row = blockIdx.x;
    if(row >= M) return;

    extern __shared__ float shared_mem[];
    float *top_arr = shared_mem + threadIdx.x * K;

    for(int i=0; i<K; i++) {
        top_arr[i] = -INFINITY;
    }
    __syncthreads();

    const float *row_data = matrix + row * N;
    for(int i = threadIdx.x; i < N; i += blockDim.x) {
        insert(top_arr, K, row_data[i]);
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if(threadIdx.x < stride) {
            float *neighbor = shared_mem + (threadIdx.x + stride) * K;
            for(int i=0; i<K; i++) {
                insert(top_arr, K, neighbor[i]);
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        float *out_row = output + row * K;
        for (int i=0; i<K; i++) {
            out_row[i] = top_arr[i];
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int K, M, N;
    std::cin >> K >> M >> N;

    K = (K*N + 99) / 100;

    float *h_mat = (float*) malloc(M*N * sizeof(float));
    float *h_out = (float*) malloc(M*K * sizeof(float));

    for(int i=0; i<M*N; i++) {
        std::cin >> h_mat[i];
    }

    float *d_mat, *d_out;
    cudaMalloc(&d_mat, M*N * sizeof(float));
    cudaMalloc(&d_out, M*K * sizeof(float));

    cudaMemcpy(d_mat, h_mat, M*N * sizeof(float), cudaMemcpyHostToDevice);

    // on-the-fly smem resizing
    // on the GB200, limited at 227KB/block
    // on all GPUs, shared memory > 48KB must be dynamic
    // reason why shared_mem[] is extern __shared__
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#id398
    // https://intro-to-cuda.readthedocs.io/en/latest/tutorial/shared_memory.html
    int threadsPerBlock;
    if(K > 400)
        threadsPerBlock = 64;
    else if(K > 200)
        threadsPerBlock = 128;
    else
        threadsPerBlock = 256;
    int smemSize = threadsPerBlock * K * sizeof(float);
    cudaFuncSetAttribute(rtopk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);

    rtopk_kernel<<<M, threadsPerBlock, smemSize>>>(d_mat, d_out, M, N, K);

    cudaMemcpy(h_out, d_out, M*K * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(3);
    for (int i=0; i<M*K; i++) {
        std::cout << h_out[i] << ((i+1) % K == 0 ? '\n' : ' ');
    }

    return 0;
}

