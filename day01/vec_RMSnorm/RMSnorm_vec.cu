#include <cuda_runtime.h>
#include "vec_utils.h"


template <typename T>
__device__ T warpReduceSum(T val) {
    for (int i = 32 / 2; i > 0; i /= 2) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}


template <typename T>
__device__ T BlockReduceSum(T val) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int warp_num = (blockDim.x + 32 - 1) / 32;
    static __shared__ T shared[64];
    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    T sum = tid < warp_num ? shared[tid] : (T)0;
    sum = warpReduceSum<T>(sum);
    return sum;
}

template <typename T>
__global__ void RMSnorm(T* input, T* scale, float eps, const int num_tokens, const int hidden_dim) {
    int vec_size = Vec<T>::size;
    using VecT = typename Vec<T>::Type;
    
    int elements_per_vec = hidden_dim / vec_size;
    float thread_sum = 0.0f;
    VecT* d_out = reinterpret_cast<VecT*>(input + blockIdx.x * hidden_dim);

    for(int i = 0; i < elements_per_vec; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if(idx < elements_per_vec) {
            VecT vec = d_out[idx];
            thread_sum += vec.x * vec.x;
            thread_sum += vec.y * vec.y;
            thread_sum += vec.z * vec.z;
            thread_sum += vec.w * vec.w;
        }
    }
    thread_sum = BlockReduceSum<float>(thread_sum);

    __shared__ float inv_mean;
    if(threadIdx.x == 0) {
        inv_mean = rsqrtf(thread_sum / hidden_dim + eps);
    }
    __syncthreads();
    VecT* para = reinterpret_cast<VecT*>(scale);
    for(int i = 0; i < elements_per_vec; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if(idx < elements_per_vec) {
            VecT vec = d_out[idx];
            VecT s = para[idx];
            
            d_out[idx].x = vec.x * inv_mean * s.x;
            d_out[idx].y = vec.y * inv_mean * s.y;
            d_out[idx].z = vec.z * inv_mean * s.z;
            d_out[idx].w = vec.w * inv_mean * s.w;
        }
    }
}  
   

template<>
__global__ void RMSnorm(half* input, half* scale, float eps, const int num_tokens, const int hidden_dim) {
    int vec_size = Vec<half>::size;
    using VecT = typename Vec<half>::Type;
    float thread_sum = 0.0f;
    VecT* d_out = reinterpret_cast<VecT*>(input + blockIdx.x * hidden_dim);

    for(int idx = 0; idx < hidden_dim / vec_size; idx += blockDim.x) {
        VecT out = d_out[idx];
        thread_sum += __half2float(out.x) * __half2float(out.x);
        thread_sum += __half2float(out.y) * __half2float(out.y);
    }
    thread_sum = BlockReduceSum<float>(thread_sum);
    
    __shared__ float inv_mean;
    if(threadIdx.x == 0) {
        inv_mean = rsqrtf(thread_sum / hidden_dim + eps);
    }
    __syncthreads();

    VecT* para = reinterpret_cast<VecT*>(scale);
    for(int idx = 0; idx < hidden_dim / vec_size; idx += blockDim.x) {
        VecT out = d_out[idx];

        d_out[idx].x = __float2half(inv_mean * __half2float(out.x) * __half2float(para[idx].x));
        d_out[idx].y = __float2half(inv_mean * __half2float(out.y) * __half2float(para[idx].y));
    }
}

template <typename T>
void launchRMSNorm(T* input, T* scale, float eps,const int num_tokens, const int hidden_dim) {
    int vec_size = Vec<T>::size;
    int num_threads = hidden_dim / vec_size;
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSnorm<T><<<grid, block>>>(input, scale, eps, num_tokens, hidden_dim);
}

template void launchRMSNorm<float>(float* input, float* scale, float eps, const int num_tokens, const int hidden_dim);

//template void launchRMSNorm<half>(half* input, half* scale, float eps, const int num_tokens, const int hidden_dim);




