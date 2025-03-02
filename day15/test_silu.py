import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# 编译CUDA扩展
# 注意：确保vec_utils.h文件存在于正确的路径
silu_and_mul = load(
    name="silu_and_mul",
    sources=["silu.cu"],
    verbose=True
)

def test_silu_and_mul():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 4
    intermedia_size = 128
    
    # 创建随机输入张量
    # 形状为 [batch_size, 2, intermedia_size]，符合CUDA函数的要求
    x = torch.randn(batch_size, 2, intermedia_size, device='cuda', dtype=torch.float32)
    
    # 运行CUDA实现
    cuda_output = silu_and_mul.silu_and_mul(x)
    
    # 使用PyTorch函数计算参考输出
    # 从输入中提取两个部分
    x_part = x[:, 0, :]  # 形状: [batch_size, intermedia_size]
    y_part = x[:, 1, :]  # 形状: [batch_size, intermedia_size]
    
    # 对x_part应用SiLU并乘以y_part
    reference_output = F.silu(x_part) * y_part
    
    # 比较结果
    max_diff = torch.max(torch.abs(cuda_output - reference_output))
    print(f"最大差异: {max_diff.item()}")
    
    # 检查输出是否足够接近
    assert torch.allclose(cuda_output, reference_output, rtol=1e-5, atol=1e-5), \
        "CUDA输出与PyTorch参考输出不匹配"
    
    print("测试通过！CUDA实现与PyTorch参考实现匹配。")
    
    # 如果可用，测试半精度
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        print("正在测试半精度...")
        
        x_half = x.half()
        cuda_output_half = silu_and_mul.silu_and_mul(x_half)
        
        # 使用半精度计算参考
        x_part_half = x_half[:, 0, :]
        y_part_half = x_half[:, 1, :]
        reference_output_half = F.silu(x_part_half) * y_part_half
        
        max_diff_half = torch.max(torch.abs(cuda_output_half - reference_output_half))
        print(f"最大差异（半精度）: {max_diff_half.item()}")
        
        # 检查输出是否足够接近，为半精度使用更高的容差
        assert torch.allclose(cuda_output_half, reference_output_half, rtol=1e-3, atol=1e-3), \
            "CUDA输出与PyTorch参考输出不匹配（半精度）"
        
        print("半精度测试通过！")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_silu_and_mul()
    else:
        print("CUDA不可用，无法运行测试。")
