//! 参考 https://github.com/2395959141/CUDA-Learn-Notes/blob/main/kernels/rms-norm/rms_norm.cu
#include <algorithm>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>
#include <iostream>

// -------------------------------------- FP32
// -------------------------------------- Warp Reduce Sum
#define WARP_SIZE 32

template<const int KwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
    #pragma unroll
    for (int i = KwarpSize >> 1; i >= 1; i >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, i);
        }
    return val;
}

template<const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_fp32(float val) {
    constexpr int NUM_WRAPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float shared[NUM_WRAPS];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    static __shared__ float shared[NUM_WRAPS];

    val = warp_reduce_sum_fp32(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
}

template<const int NUM_THREADS = 256>
__global__ void rms_norm_fp32_kernel(float* x, float* y, float N, float K, float g) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    //! 定义分母添加的常数
    const float epsilon = 1e-5f;

    //! 这里一个block处理一行，所以将 variance 存储在共享内存中
    __shared__ float s_variance;
    float value = (idx < N * K) ? x[idx] : 0.0f;  // load once only
    //! 开始求每一行的归一化因子
    float variance = value * value;
    variance = block_reduce_sum_fp32<NUM_THREADS>(variance); 
    if (tid == 0) {
        s_variance = rsqrt(variance / (float)K + epsilon);
    }
    __syncthreads();
    
    if (idx < N * K) {
        y[idx] = value * s_variance * g;
    }
}



// -------------------------------------- FP32 * 4
// -------------------------------------- Warp Reduce Sum
//! fp32 * 4 的情况下，一个线程处理4个value。所以一个block中的线程除以4

#define FLOAT4(value) (reinterpret_cast<float4*>(&value)[0])

template<const int NUM_THREADS = 256 / 4>
__global__ void rms_norm_fp32_4_kernel(float* x, float* y, float N, float K, float g) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;
    //! 添加一个float4变量用于存储4个value
    //! 因为没有对 4个float 进行向量化计算的指令，所以需要展开计算
    float4 reg_x = FLOAT4(x[idx]);
    float variance = (idx < N * K) ? (
        reg_x.x * reg_x.x + reg_x.y * reg_x.y + 
        reg_x.z * reg_x.z + reg_x.w * reg_x.w
    ) : 0.0f;

    variance = block_reduce_sum_fp32<NUM_THREADS>(variance);
    if (tid == 0) {
        s_variance = rsqrt(variance / (float)K + epsilon);
    }
    __syncthreads();

    //! 将4个value 写入到y中
    //!  因为没有对 4个float 进行向量化计算的指令，所以需要展开计算
    float4 reg_y = FLOAT4(y[idx]);
    if (idx < N * K) {
        reg_y.x = reg_x.x * s_variance * g;
        reg_y.y = reg_x.y * s_variance * g;
        reg_y.z = reg_x.z * s_variance * g;
        reg_y.w = reg_x.w * s_variance * g;
        y[idx] = reg_y;
    }
}


// -------------------------------------- 一些检查的宏定义
// -------------------------------------- 
#define CHECK_TORCH_TENSOR_DTYPE(T, th_dype) \
    if ((T).options().dtype() != (th_dype)) { \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("Tensor dtype mismatch"); \
    }


#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                        \
    assert((T1).dim() == (T2).dim());                           \
    for (int i = 0; i < (T1).dim(); i++) {                      \
        if ((T1).size(i) != (T2).size(i)) {                     \
            throw std::runtime_error("Tensor shape mismatch");  \
        }                                                       \
    }                                                           \


#define LAUNCH_RMS_NORM_FP32_KERNEL(K)                            \
    rms_norm_fp32_kernel<(K)>                                     \
        <<<grid, block>>>(reinterpert_cast<float*>(x.data_ptr()), \
                          reinterpert_cast<float*>(y.data_ptr()), \
                          N, (K), g);                                        



#define DISPATCH_RMS_NORM_FP32_KERNEL(N, K)                         \
    dim3 block((K));                                                \
    dim3 grid((N));                                                 \
                                                                    \
    switch ((K)) {                                                  \
    case 64:                                                        \
        LAUNCH_RMS_NORM_FP32_KERNEL(64);                            \
        break;                                                      \
    case 128:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(128);                           \
        break;                                                      \
    case 256:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(256);                           \
        break;                                                      \
    case 512:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(512);                           \
        break;                                                      \
    case 1024:                                                      \
        LAUNCH_RMS_NORM_FP32_KERNEL(1024);                          \
        break;                                                      \
    default:                                                        \
        throw std::runtime_error("Only support K: 64/128/256/512/1024"); \
        break;                                                      \
    }



#define LAUNCH_RMS_NORM_FP32x4_KERNEL(K) \
    rms_norm_fp32_4_kernel<K / 4> \
        <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                          reinterpret_cast<float*>(y.data_ptr()), \
                          N, (K), g);

#define DISPATCH_RMS_NORM_FP32x4_KERNEL(N, K)                         \
    dim3 block((K));                                                \
    dim3 grid((N));                                                 \
                                                                    \
    switch ((K)) {                                                  \
    case 64:                                                        \
        LAUNCH_RMS_NORM_FP32_KERNEL(64);                            \
        break;                                                      \
    case 128:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(128);                           \
        break;                                                      \
    case 256:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(256);                           \
        break;                                                      \
    case 512:                                                       \
        LAUNCH_RMS_NORM_FP32_KERNEL(512);                           \
        break;                                                      \
    case 1024:                                                      \
        LAUNCH_RMS_NORM_FP32_KERNEL(1024);                          \
        break;                                                      \
    default:                                                        \
        throw std::runtime_error("Only support K: 64/128/256/512/1024"); \
        break;                                                      \
    }



void rms_norm_fp32(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(x, y)

    int N = x.size(0);
    int K = x.size(1);
    DISPATCH_RMS_NORM_FP32_KERNEL(N, K);
}

void rms_norm_fp32x4(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(x, y)

    int N = x.size(0);
    int K = x.size(1);
    DISPATCH_RMS_NORM_FP32x4_KERNEL(N, K);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_fp32, "RMS Normalization (CUDA)");
    m.def("rms_norm_fp32x4", &rms_norm_fp32x4, "RMS Normalization (CUDA)");
}