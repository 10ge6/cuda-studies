#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_LEN 256

struct Query {
    int L, C1, C2;
    int offset;
};

__global__ void softmax_kernel(const float* matrix, float* output, const Query* queries, int N) {
    int query_it = blockIdx.x;
    Query q = queries[query_it];

    int len = q.C2 - q.C1 + 1;
    const float* input = matrix + q.L * N + q.C1;
    float* out = output + q.offset;

    int tid = threadIdx.x;
    int warp_id = tid/32;
    __shared__ float reduction[BLOCK_LEN/32];

    float maxval = -INFINITY;

    bool can_f4 = ((q.L * N + q.C1) % 4 == 0) &&  // (arbitrary query boundaries necessitate this)
                  (q.offset % 4 == 0);            // 16-byte aligned, OK to vectorize memory access
    int vec_end = (len / 4) * 4;

    if(can_f4) {
        for (int i = tid; i < len/4; i += BLOCK_LEN) {
            float4 val = reinterpret_cast<const float4*>(&input[i*4])[0];
            maxval = fmaxf(maxval, val.x);
            maxval = fmaxf(maxval, val.y);
            maxval = fmaxf(maxval, val.z);
            maxval = fmaxf(maxval, val.w);
        }
        for (int i = vec_end + tid; i < len; i += BLOCK_LEN) {
            maxval = fmaxf(maxval, input[i]);
        }
    } else {
        for (int i = tid; i < len; i += BLOCK_LEN) {
            maxval = fmaxf(maxval, input[i]);
        }
    }

    for (int mask = 16; mask > 0; mask /= 2) {
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if (tid % 32 == 0) {
        reduction[warp_id] = maxval;
    }
    __syncthreads();

    if (warp_id == 0) {
        maxval = tid < BLOCK_LEN/32 ? reduction[tid] : -INFINITY;
        for (int mask = 16; mask > 0; mask /= 2) {
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if (tid == 0) {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];

    float divisor = 0.f;

    if(can_f4) {
        for (int i = tid; i < len/4; i += BLOCK_LEN) {
            float4 val = reinterpret_cast<const float4*>(&input[i*4])[0];
            divisor += __expf(val.x - maxval);
            divisor += __expf(val.y - maxval);
            divisor += __expf(val.z - maxval);
            divisor += __expf(val.w - maxval);
        }
        for (int i = vec_end + tid; i < len; i += BLOCK_LEN) {
            divisor += __expf(input[i] - maxval);
        }
    } else {
        for (int i = tid; i < len; i += BLOCK_LEN) {
            divisor += __expf(input[i] - maxval);
        }
    }

    for (int mask = 16; mask > 0; mask /= 2) {
        divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (tid % 32 == 0) {
        reduction[warp_id] = divisor;
    }
    __syncthreads();

    if (warp_id == 0) {
        divisor = tid < BLOCK_LEN/32 ? reduction[tid] : 0;
        for (int mask = 16; mask > 0; mask /= 2) {
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (tid == 0) {
        reduction[0] = divisor;
    }
    __syncthreads();
    divisor = reduction[0];

    if(can_f4) {
        for (int i = tid; i < len/4; i += BLOCK_LEN) {
            float4 val = reinterpret_cast<const float4*>(&input[i*4])[0];
            val.x = __expf(val.x - maxval) / divisor;
            val.y = __expf(val.y - maxval) / divisor;
            val.z = __expf(val.z - maxval) / divisor;
            val.w = __expf(val.w - maxval) / divisor;
            reinterpret_cast<float4*>(&out[i*4])[0] = val;
        }
        for (int i = vec_end + tid; i < len; i += BLOCK_LEN) {
            out[i] = __expf(input[i] - maxval) / divisor;
        }
    } else {
        for (int i = tid; i < len; i += BLOCK_LEN) {
            out[i] = __expf(input[i] - maxval) / divisor;
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int M, N;
    std::cin >> M >> N;

    int sz = M*N * sizeof(float);

    float *h_m = (float*) malloc(sz);

    for(int i=0; i<M*N; i++)
        std::cin >> h_m[i];

    std::vector<Query> h_queries;
    int count = 0;

    int L, C1, C2;
    while(std::cin >> L >> C1 >> C2) {
        int len = C2 - C1 + 1;
        h_queries.push_back({L, C1, C2, count});
        count += len;
    }

    float *d_m, *ans;
    cudaMalloc((void **) &d_m, sz);
    cudaMalloc((void **) &ans, count * sizeof(float));

    cudaMemcpy(d_m, h_m, sz, cudaMemcpyHostToDevice);

    Query* d_queries;
    float *out = (float*) malloc(count * sizeof(float));
    cudaMalloc((void **) &d_queries, h_queries.size() * sizeof(Query));

    cudaMemcpy(d_queries, h_queries.data(), h_queries.size() * sizeof(Query), cudaMemcpyHostToDevice);

    softmax_kernel<<<h_queries.size(), BLOCK_LEN>>>(d_m, ans, d_queries, N);

    cudaDeviceSynchronize();

    cudaMemcpy(out, ans, count * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(4);
    for(const Query& q : h_queries) {
        int len = q.C2 - q.C1 + 1;
        for(int i = 0; i < len; i++) {
          std::cout << out[q.offset + i] << (i < len-1 ? ' ' : '\n');
        }
    }

    return 0;
}

