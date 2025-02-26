import torch
import os
from torch.utils.cpp_extension import load

# 编译CUDA扩展
current_dir = os.path.dirname(os.path.abspath(__file__))
concat_kv_module = load(
    name="concat_kv",
    sources=[os.path.join(current_dir, "concat_kv.cu")],
    verbose=True
)

def test_concat_kv_cache():
    """测试KV缓存拼接功能的正确性"""
    # 设置测试参数
    batch_size = 2
    kv_head_num = 4
    max_q_len = 8
    head_size = 64
    max_seq_len = 1024
    num_layers = 3
    layer_id = 1  # 测试第二层
    
    # 当前查询和历史长度
    cur_query_length = torch.tensor([5, 6], dtype=torch.int32, device="cuda")  # 第一个批次5个token，第二个批次6个token
    history_length = torch.tensor([10, 20], dtype=torch.int32, device="cuda")  # 第一个批次历史10个token，第二个批次20个token
    
    # 创建输入KV张量
    k_src = torch.randn(batch_size, kv_head_num, max_q_len, head_size, 
                        dtype=torch.float16, device="cuda")
    v_src = torch.randn(batch_size, kv_head_num, max_q_len, head_size, 
                        dtype=torch.float16, device="cuda")
    
    # 创建KV缓存张量
    k_dst = torch.zeros(num_layers, batch_size, kv_head_num, max_seq_len, head_size, 
                        dtype=torch.float16, device="cuda")
    v_dst = torch.zeros(num_layers, batch_size, kv_head_num, max_seq_len, head_size, 
                        dtype=torch.float16, device="cuda")
    
    # 调用拼接函数
    concat_kv_module.concat_kv_cache(
        k_src, v_src, layer_id, cur_query_length, history_length, k_dst, v_dst
    )
    
    # 验证结果
    print("检查拼接结果是否正确...")
    
    # 验证第一个批次
    for b in range(batch_size):
        history_len = history_length[b].item()
        query_len = cur_query_length[b].item()
        
        # 检查key缓存
        for h in range(kv_head_num):
            for t in range(query_len):
                # 原始数据
                src_key = k_src[b, h, t, :].cpu()
                # 目标缓存中的结果
                dst_key = k_dst[layer_id, b, h, history_len + t, :].cpu()
                
                # 检查是否匹配
                if not torch.allclose(src_key, dst_key):
                    print(f"错误：批次{b}，头{h}，token{t}的key不匹配")
                    return False
                
                # 同样检查value
                src_value = v_src[b, h, t, :].cpu()
                dst_value = v_dst[layer_id, b, h, history_len + t, :].cpu()
                
                if not torch.allclose(src_value, dst_value):
                    print(f"错误：批次{b}，头{h}，token{t}的value不匹配")
                    return False
    
    # 检查其他层是否未受影响
    for other_layer in range(num_layers):
        if other_layer == layer_id:
            continue
        
        if torch.sum(torch.abs(k_dst[other_layer])) > 0:
            print(f"错误：层{other_layer}的key缓存被意外修改")
            return False
        
        if torch.sum(torch.abs(v_dst[other_layer])) > 0:
            print(f"错误：层{other_layer}的value缓存被意外修改")
            return False
    
    print("测试通过！KV缓存拼接功能正确")
    return True

def visualize_kv_cache(k_dst, v_dst, layer_id, batch_id=0):
    """可视化KV缓存的非零部分"""
    k_layer = k_dst[layer_id, batch_id]
    v_layer = v_dst[layer_id, batch_id]
    
    # 计算每个位置的KV向量的L2范数
    k_norms = torch.norm(k_layer, dim=2)  # [kv_head_num, max_seq_len]
    v_norms = torch.norm(v_layer, dim=2)  # [kv_head_num, max_seq_len]
    
    print(f"Layer {layer_id}, Batch {batch_id} 的KV缓存非零位置:")
    print("Key缓存非零位置:")
    non_zero_k = (k_norms > 0).nonzero()
    print(non_zero_k)
    
    print("Value缓存非零位置:")
    non_zero_v = (v_norms > 0).nonzero()
    print(non_zero_v)
    
    # 打印几个样本值
    if len(non_zero_k) > 0:
        head_idx, seq_pos = non_zero_k[0]
        print(f"Key缓存样本 (head={head_idx}, pos={seq_pos}):")
        print(k_layer[head_idx, seq_pos, :10])  # 只打印前10个元素
    
    if len(non_zero_v) > 0:
        head_idx, seq_pos = non_zero_v[0]
        print(f"Value缓存样本 (head={head_idx}, pos={seq_pos}):")
        print(v_layer[head_idx, seq_pos, :10])  # 只打印前10个元素

if __name__ == "__main__":
    # 运行测试
    success = test_concat_kv_cache()
    
    if success:
        # 创建更大的测例进行性能测试
        print("\n开始性能测试...")
        batch_size = 4
        kv_head_num = 32
        max_q_len = 32
        head_size = 128
        max_seq_len = 2048
        num_layers = 32
        layer_id = 5
        
        cur_query_length = torch.tensor([16, 24, 32, 8], dtype=torch.int32, device="cuda")
        history_length = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device="cuda")
        
        k_src = torch.randn(batch_size, kv_head_num, max_q_len, head_size, 
                            dtype=torch.float16, device="cuda")
        v_src = torch.randn(batch_size, kv_head_num, max_q_len, head_size, 
                            dtype=torch.float16, device="cuda")
        
        k_dst = torch.zeros(num_layers, batch_size, kv_head_num, max_seq_len, head_size, 
                            dtype=torch.float16, device="cuda")
        v_dst = torch.zeros(num_layers, batch_size, kv_head_num, max_seq_len, head_size, 
                            dtype=torch.float16, device="cuda")
        
        # 预热
        for _ in range(5):
            concat_kv_module.concat_kv_cache(
                k_src, v_src, layer_id, cur_query_length, history_length, k_dst, v_dst
            )
        
        # 计时
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            concat_kv_module.concat_kv_cache(
                k_src, v_src, layer_id, cur_query_length, history_length, k_dst, v_dst
            )
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / 10
        print(f"平均执行时间: {elapsed_time:.3f} ms")
        
        # 可视化结果
        visualize_kv_cache(k_dst, v_dst, layer_id, batch_id=0)

