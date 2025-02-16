#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__device__ T warpReduceSum(T val) {
    for (int i = 32 / 2; i > 0; i /= 2) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}


template <typename T>
__device__ T BlockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    T sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    sum = warpReduceSum<T>(sum);
    return sum;
}

    
template <typename T> 
__global__ void RMSnorm(T* input, T* scale, const int num_tokens, const int hidden_dim) {
    // 每个token独立处理
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr float eps = 1e-5f;
    
    // 共享内存用于存储平方和
    __shared__ float s_mean;
    float thread_sum = 0.0f;

    // 每个线程计算部分元素的平方和
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input[token_idx * hidden_dim + i];
        thread_sum += val * val;
    }

    // 块内归约求和
    thread_sum = BlockReduceSum<float>(thread_sum);
    
    // 计算归一化系数
    if (tid == 0) {
        s_mean = rsqrtf(thread_sum / hidden_dim + eps);
    }
    __syncthreads();

    // 应用归一化和缩放
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input[token_idx * hidden_dim + i];
        input[token_idx * hidden_dim + i] = val * s_mean * scale[i];
    }
}




    
    
    
