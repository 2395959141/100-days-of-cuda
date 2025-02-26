#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__half* to_half_ptr(at::Half* ptr) {
    return reinterpret_cast<__half*>(ptr);
}

const __half* to_half_ptr(const at::Half* ptr) {
    return reinterpret_cast<const __half*>(ptr);
}

__global__ void append_key_cache(__half *k_dst, // [num layers, bs, kv head num, max_q_len, head size]
                                 const size_t layer_offset,
                                 const __half *k_src, // [bs, kv_head num, max_q_len, head size]
                                 const int kv_head_num,
                                 const int head_size,
                                 const int *cur_query_length,
                                 const int *history_length,
                                 const int max_q_len,
                                 const int max_seq_len)
{
    //! kv_cache 【层数，batch_size, 注意力头数，max_seq_len, head_size】
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;

    //* 指针偏移到当前layer的k cache
    __half *k_cache_dst = k_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id]; //* 第batchid个输入的长度
    int cumsum_seq_len = history_length[batch_id];

    // note: the if judge is a must, because the max_q_len is GTE than cur_seq_len.
    if (token_id < cur_seq_len) //! 不超过当前输入长度
    {
        // [batch, head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len + max q len], head size]
        //! kv的偏移
        int src_offset = batch_id * kv_head_num * max_q_len * head_size +  
                         head_id * max_q_len * head_size +
                         token_id * head_size + tid;

        //! kv cache的偏移
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size + //* 移动到batch起始位置
                         head_id * max_seq_len * head_size +                //* 移动到head起始位置
                         (cumsum_seq_len + token_id) * head_size + tid;     //* 移动到token起始位置

        k_cache_dst[dst_offset] = k_src[src_offset]; //! kv cache
    }
}

__global__ void append_value_cache(__half* v_dst,
                                   const size_t layer_offset,
                                   const __half* v_src,
                                   const int kv_head_nums,
                                   const int head_size,
                                   const int *cur_query_length,
                                   const int *history_length,
                                   const int max_q_length,
                                   const int max_seq_len)
{
    //! kv_cache 【层数，batch_size, 注意力头数，max_seq_len, head_size】
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;

    __half* v_cache_dst = v_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];

    if (token_id < cur_seq_len)
    {
        int src_offset = batch_id * kv_head_nums * max_q_length * head_size +
                         head_id * max_q_length * head_size +
                         token_id * head_size + tid;

        int dst_offset = batch_id * kv_head_nums * max_seq_len * head_size +
                         head_id * max_seq_len * head_size +
                         (cumsum_seq_len + token_id) * head_size + tid;

        v_cache_dst[dst_offset] = v_src[src_offset];
    }
}

void launchConcatKVCache(
    torch::Tensor k_src,
    torch::Tensor v_src,
    int layer_id,
    torch::Tensor cur_query_length,
    torch::Tensor history_length,
    torch::Tensor k_dst,
    torch::Tensor v_dst) 
{
    // 类型检查
    TORCH_CHECK(k_src.scalar_type() == at::ScalarType::Half, 
               "输入张量k_src必须是Half类型");
    TORCH_CHECK(v_src.scalar_type() == at::ScalarType::Half,
               "输入张量v_src必须是Half类型");
    
    // 设备检查  
    TORCH_CHECK(k_src.device().is_cuda(), "输入张量k_src必须在CUDA设备上");
    TORCH_CHECK(v_src.device().is_cuda(), "输入张量v_src必须在CUDA设备上");
    
    // 边界检查
    TORCH_CHECK(layer_id >= 0 && layer_id < k_dst.size(0), 
               "layer_id超出有效范围");
    
    // 从张量获取维度信息
    const int batch_size = k_src.size(0);
    const int kv_head_num = k_src.size(1);
    const int max_q_len = k_src.size(2);
    const int head_size = k_src.size(3);
    const int max_seq_len = k_dst.size(3);
    
    // 计算层偏移量
    const size_t layer_offset = layer_id * batch_size * kv_head_num * max_seq_len * head_size;
    
    // 设置CUDA执行配置
    const int blockSize = head_size;
    const dim3 grid(max_q_len, batch_size, kv_head_num);
    
    // 使用类型转换
    append_key_cache<<<grid, blockSize>>>(
        to_half_ptr(k_dst.data_ptr<at::Half>()),
        layer_offset,
        to_half_ptr(k_src.data_ptr<at::Half>()),
        kv_head_num,
        head_size,
        cur_query_length.data_ptr<int>(),
        history_length.data_ptr<int>(),
        max_q_len,
        max_seq_len);

    append_value_cache<<<grid, blockSize>>>(
        to_half_ptr(v_dst.data_ptr<at::Half>()),
        layer_offset,
        to_half_ptr(v_src.data_ptr<at::Half>()),
        kv_head_num,
        head_size,
        cur_query_length.data_ptr<int>(),
        history_length.data_ptr<int>(),
        max_q_len,
        max_seq_len);
        
    // 可选：检查CUDA错误
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA错误: ", cudaGetErrorString(error));
}

// PyBind绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("concat_kv_cache",
          &launchConcatKVCache,
          "Concatenate key-value cache (CUDA)");
} 