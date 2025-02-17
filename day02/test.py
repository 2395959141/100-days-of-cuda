import torch
from torch.utils.cpp_extension import load
import time
import os
import matplotlib.pyplot as plt

sources = [
    'binding.cpp',
    'FP16_GELU.cu'
]

# try:
FP16_GELU = load(
        name="fp16_gelu_ext",
        sources=sources,
        verbose=True,
    )
    # if not hasattr(FP16_GELU, 'FP16_GELU'):
    #      raise RuntimeError("FP16_GELU extension loaded but FP16_GELU function not found")
# except Exception as e:
#     print(f"Error loading FP16_GELU extension: {e}")
#     raise e

print("Custom CUDA FP16_GELU extension loaded successfully.")

def pytorch_gelu(x):
    """PyTorch原生实现的GELU（与CUDA实现严格对齐）"""
    # 使用与CUDA实现完全相同的常数
    alpha = 0.7978845608028654  # sqrt(2/pi) 的精确值
    beta = 0.044714998453855515  # CUDA实现中的beta值
    
    # 严格对齐CUDA实现的计算步骤：
    # 1. 计算x的三次方
    x_cubed = x ** 3
    # 2. 计算内部多项式项（保持运算顺序一致）
    inner = beta * x_cubed
    # 3. 计算tanh的输入（保持乘法顺序一致）
    tanh_input = alpha * x * (1 + inner)
    # 4. 应用tanh激活
    tanh_out = torch.tanh(tanh_input)
    # 5. 组合最终结果（保持运算顺序一致）
    return 0.5 * x * (1 + tanh_out)

# 在测试前添加确定性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 使用固定随机种子
torch.manual_seed(42)

# 修改测试尺寸
tensor_sizes = [
    (4096, 512),    # 512是2的倍数
    (4096, 1024),   # 1024是2的倍数
    (4096, 2048),   # 2048是2的倍数
    (4096, 4096),   # 4096是2的倍数
]

# 添加两个列表来存储性能数据
pytorch_times = []
custom_times = []
sizes = []

for size in tensor_sizes:
    print("=" * 50)
    print(f"Testing size: {size}")
    print("=" * 50)
    
    # 修改后确保内存对齐
    def create_aligned_tensor(shape, alignment=128):
        numel = torch.Size(shape).numel()
        aligned_numel = ((numel + alignment - 1) // alignment) * alignment
        buffer = torch.empty(aligned_numel, device='cuda', dtype=torch.float16)
        return buffer[:numel].view(shape)

    input_tensor = create_aligned_tensor(size)
    
    num_tokens, hidden_dim = input_tensor.shape  # 自动获取维度
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Num tokens: {num_tokens}")
    print(f"Hidden dim: {hidden_dim}")
    
    # 在测试循环中添加：
    assert input_tensor.dim() == 2, "输入必须是2D张量"
    assert input_tensor.is_contiguous(), "输入张量必须是连续的"

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        result_pytorch = pytorch_gelu(input_tensor)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 5

     # 自定义CUDA实现基准测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        result_custom = FP16_GELU.FP16_GELU8(input_tensor)
        if result_custom is None:
            raise RuntimeError("CUDA function returned None")
    torch.cuda.synchronize()
    custom_time = (time.time() - start) / 5

    # 结果对比
    max_diff = torch.max(torch.abs(result_pytorch - result_custom)).item()
    print(f"PyTorch time: {pytorch_time:.6f}s")
    print(f"Custom CUDA time: {custom_time:.6f}s")
    print(f"Speedup: {pytorch_time / custom_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    # 数值正确性验证（考虑浮点误差）
    if torch.allclose(result_pytorch, result_custom, atol=1e-4):
        print("✅ Results match!")
    else:
        print("❌ Results differ!")
    
    print("=" * 50 + "\n")

     # 存储性能数据
    pytorch_times.append(pytorch_time)
    custom_times.append(custom_time)
    sizes.append(size[0] * size[1])  # 存储元素总数

if __name__ == "__main__":
    # 在代码末尾添加以下绘图代码
    plt.figure(figsize=(10, 6))

    # 绘制 PyTorch 实现的时间曲线
    plt.plot([size[1] for size in tensor_sizes], pytorch_times, 'o-', label='PyTorch GELU', color='blue')

    # 绘制自定义 CUDA 实现的时间曲线
    plt.plot([size[1] for size in tensor_sizes], custom_times, 'o-', label='Custom CUDA GELU', color='orange')

    # 添加图表元素
    plt.title('PyTorch vs Custom CUDA GELU Performance Comparison')
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which="both", ls="-")
    plt.legend()

    # 保存图表
    plt.savefig('./rmsnorm_performance_comparison.png')

    # 显示图表
    plt.show()









