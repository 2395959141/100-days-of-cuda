#include <cstdint>  // 添加头文件
#include <cassert>
#include <cuda_runtime.h>

// 在文件开头添加SumOp的定义
template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

//* 将Reduce进行模板类，根据模板参数支持Reduce的各种操作
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
// 把block reduce拆分为多个warp reduce来计算
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    // 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ float warpres[64];
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L65用0号线程写入warp res
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    float warp_val = tid < warp_nums ? warpres[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}

// 一个blk计算一个元素
// mat * vec = {M, N} * {N, 1}/{1, N}
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int rows, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float thread_local_sum = 0.0f;
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        // 修正后的索引计算
        float4 mat4 = reinterpret_cast<float4*>(matrix)[
            bid * (cols / VEC_SIZE) + i * blockDim.x + tid
        ];
        float4 vec4 = reinterpret_cast<float4*>(vector)[
            i * blockDim.x + tid
        ];
        
        thread_local_sum += mat4.x * vec4.x;
        thread_local_sum += mat4.y * vec4.y;
        thread_local_sum += mat4.z * vec4.z;
        thread_local_sum += mat4.w * vec4.w;
    }
    // reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    // store to gmem
    if(tid == 0) {
        res[bid] = reduce_res;
    }
    __syncthreads();
}

template<int VECS_PER_THREAD, int VEC_SIZE>
void launchGEMV(float* matrix, float* vector, float* res, int rows, int cols) {
    const int block_size = 256;
    const int elements_per_block = block_size * VEC_SIZE * VECS_PER_THREAD;
    
    assert((cols % elements_per_block) == 0 && 
        "cols must be divisible by (block_size * VEC_SIZE * VECS_PER_THREAD)");
    
    assert(reinterpret_cast<uintptr_t>(matrix) % 16 == 0 && "Matrix must be 16-byte aligned");
    assert(reinterpret_cast<uintptr_t>(vector) % 16 == 0 && "Vector must be 16-byte aligned");
    
    gemv<VECS_PER_THREAD, VEC_SIZE><<<rows, block_size>>>(matrix, vector, res, rows, cols);
}

// 修改模板实例化部分，添加显式命名
template<>
void launchGEMV<1,4>(float* matrix, float* vector, float* res, int rows, int cols) {
    gemv<1,4><<<rows, 256>>>(matrix, vector, res, rows, cols);
}

template<>
void launchGEMV<2,4>(float* matrix, float* vector, float* res, int rows, int cols) {
    gemv<2,4><<<rows, 256>>>(matrix, vector, res, rows, cols);
}

template<>
void launchGEMV<4,4>(float* matrix, float* vector, float* res, int rows, int cols) {
    gemv<4,4><<<rows, 256>>>(matrix, vector, res, rows, cols);
}

// // 添加函数声明
// extern "C" {
//     void GEMV_1_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res);
//     void GEMV_2_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res);
//     void GEMV_4_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res);
// }

