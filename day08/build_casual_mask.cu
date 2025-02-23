#pragma once
#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_fp16.h>

//! mask 矩阵的维度为 [batch_size, max_q_len, max_k_len]
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask, //* 返回的 mask 矩阵
                            const int* q_lens,  //* 输入 [batch_size]
                            const int* k_lens,  //* 输入 [batch_size]
                            int max_q_len, //* max(q_lens) 
                            int max_k_len) //* max(k_lens) 

{
    int tid  = threadIdx.x;
    int qlen = q_lens[blockIdx.x]; //! 每个block处理一个batch，q_len有batch_size个长度，所以q_lens可以用batch_size的索引
    int klen = k_lens[blockIdx.x];

    mask += blockIdx.x * max_q_len * max_k_len; //! 得到每个block处理mask的起始位置

    while (tid < max_q_len * max_k_len){
        int q = tid / max_k_len;
        int k = tid % max_k_len;
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen); //! 考虑repeat kv和当前context
        mask[tid] = is_one ? static_cast<T>(1) : static_cast<T>(0);

        tid += blockDim.x;
    }
}                      

template<typename T>
void launchBuildCausalMasks(T* mask,
                            const int* q_lens,
                            const int* k_lens,
                            int batch_size,
                            int max_q_len,
                            int max_k_len)
{
    BuildCausalMasksConsideringContextPastKV<<<batch_size, 256>>>(mask, q_lens, k_lens, max_q_len, max_k_len);
}


template void launchBuildCausalMasks<float>(float*, const int*, const int*, int, int, int);
template void launchBuildCausalMasks<half>(half*, const int*, const int*, int, int, int);
