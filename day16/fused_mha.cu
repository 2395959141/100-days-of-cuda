#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <torch/extension.h>
#include "../day01/vec_RMSnorm/vec_utils.h"


template <typename T>
__device__ T warpReduceSum(T val) {
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__device__ T blockReduceSum(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 32;
    int warp_nums = (blockDim.x + 31) / 32;

    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);

    if (lane_id == 0) {
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);
}

template <typename T>
__device__ T warpReduceMax(T val) {
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename T>
__device__ T blockReduceMax(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 32;
    int warp_nums = (blockDim.x + 31) / 32;

    static __shared__ T warpmax[64];
    val = warpReduceMax<T>(val);

    if (lane_id == 0) {
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax<T>(warp_val);
}


//! block and thread allocation
//* 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size

//! 1 grid -> bs * num heads
//* q; input vec [bs, q num heads, 1, head size]
//* k; input vec [bs, kv num heads, 1, head size]
//* v; input vec [bs, num heads, 1, head size]

//* k_cache; output,[num_layers, bs, kv num heads, max_seq_len, head size] from prompt phase
//* v_cache; output,[num_layers, bs, kv num heads, max_seq_len, head size] from prompt phase

template<typename T>
__global__ void masked_MHA_kernel(T* q,
                T* k,
                T* v,
                T* qkv_bias,  //? llama2中ffn没有blas, chatglm中ffn有blas
                T* k_cache,
                T* v_cache,
                T* mha_output,
                const int batch_size,
                const int head_num,
                const int kv_head_num,
                const int max_seq_len,
                const int head_size,
                const int step) {
    //!  dim3 grid(head_num * batch_size);
    //! dim3 block(head_size); //vec size = 4 for fp32
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;

    //! 计算q,k,v的偏移量
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    int vec_size = Vec<T>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;

    //! kv cache的offset [从kv cache中取k,v 需要乘上 max_seq_len]
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                      kv_head_id * max_seq_len * head_size +
                      step * head_size;
    int step_stride = head_size;
    float scale = rsqrt(float(head_size));

    //! (1) 从缓存中加载 Q, K ,V
    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    const T* q_mem = q;
    const T* k_mem = k;
    const T* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec]));
        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec]));
        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec]));
    }

    extern __shared__ char sqk[];  //! 使用动态shared memory，需要添加extern

    //! sq_scalar指向共享内存起始地址，用于存储Q向量（head_size个T类型元素）
    //! logits从sq_scalar + head_size开始，存储softmax前的注意力分数（step个float元素）
    T* sq_scalar = reinterpret_cast<T*>(sqk); //* q存在smem的必要性:在第step行把q存进smem，之前的step-1行可以直接从smem load to reg参与计算
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size); //* logits存在smem的必要性:所有线程reduce的结果存到logits，需要smem
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();

    //! (2) 计算logits
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);

    //! step 表示当前要处理的token位置
    //! iter 表示当前处理的历史token位置
    for (int iter = 0; iter < step; iter++) {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_f4;

        //! (5)当iter = step - 1时，更新kv cache，并从输入kvec中获取k
        if (iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }
        //! (2)计算历史token的QK点积
        Vec_t qk = zero_f4;
        qk.x = (tid * vec_size < head_size) ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;
        qk.y = (tid * vec_size < head_size) ? sq[tid].y * kvec_qk.y * scale_f4.y : zero;
        qk.z = (tid * vec_size < head_size) ? sq[tid].z * kvec_qk.z * scale_f4.z : zero;
        qk.w = (tid * vec_size < head_size) ? sq[tid].w * kvec_qk.w * scale_f4.w : zero;
        T qk_acc = qk.x + qk.y + qk.z + qk.w;
        T attn_score = blockReduceSum<T>(qk_acc);
        if (tid == 0) {
            logits[iter] = attn_score; //! 当前Q向量与第iter个K向量的完整点积
        }
        __syncthreads();
    }
    
    //! (3) 计算Softmax归一化
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;

    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();

    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;

    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0) {
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();

    if (tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();

    //!  (4) 计算logits*V = qk * v =  [bs, num heads, 1, step] * [bs, kv num heads, step, head size]
    if (tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);
        for (int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);

            //! (5)当iter = step - 1时，更新v cache
            if (iter == step - 1) {
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
            }

            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
        }
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }
}



//! block and thread allocation
//* 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size

//! 1 grid -> bs * num heads
//* q; input vec [bs, q num heads, 1, head size]
//* k; input vec [bs, kv num heads, 1, head size]
//* v; input vec [bs, num heads, 1, head size]

//* k_cache; output,[num_layers, bs, kv num heads, max_seq_len, head size] from prompt phase
//* v_cache; output,[num_layers, bs, kv num heads, max_seq_len, head size] from prompt phase

// 封装后的启动函数
void launchDecoderMaskedMHA(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &qkv_bias,
    torch::Tensor &k_cache,
    torch::Tensor &v_cache,
    torch::Tensor &output) {
    
    // 检查输入张量是否在CUDA上
    TORCH_CHECK(q.device().is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "v must be a CUDA tensor");
    
    // 获取张量维度信息
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_size = q.size(3);
    
    // 计算CUDA网格和块大小
    dim3 blocks(seq_len, batch_size * num_heads);
    dim3 threads(head_size);
    
    // 计算共享内存大小（head_size * sizeof(float) + step * sizeof(float)）
    size_t smem_size = head_size * sizeof(float) + seq_len * sizeof(float);
    
    // 修正kernel名称和参数
    masked_MHA_kernel<float><<<blocks, threads, smem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        qkv_bias.data_ptr<float>(),  // qkv_bias (根据实际情况调整)
        k_cache.data_ptr<float>(),  // k_cache (需要补充参数)
        v_cache.data_ptr<float>(),  // v_cache (需要补充参数)
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        num_heads,  // kv_head_num (假设num_heads=kv_head_num)
        seq_len,    // max_seq_len
        head_size,
        0           // step
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error:", cudaGetErrorString(err));
}

// PyBind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decoder_masked_mha", &launchDecoderMaskedMHA, 
          "Compute decoder masked multi-head attention (CUDA)");
} 
