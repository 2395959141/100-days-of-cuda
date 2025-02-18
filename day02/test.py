import torch
from torch.utils.cpp_extension import load
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

sources = [
    'binding.cpp',
    'FP16_GELU.cu'
]

# try:
FP16_GELU = load(
        name="fp16_gelu_ext",
        sources=sources,
        verbose=True,
        extra_cuda_cflags=[
            '-O3',
            '-use_fast_math',
        ]
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

def pytorch_gelu_official(x):
    """使用PyTorch提供的GELU实现（与CUDA实现严格对齐）"""
    # 直接使用PyTorch的GELU函数
    return F.gelu(x)

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

# 在初始化列表后添加带宽存储列表
pytorch_bandwidth = []
custom_bandwidth = []

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

    # 添加预热过程
    print("Warming up...")
    # PyTorch预热
    for _ in range(10):  # 预热10次
        _ = pytorch_gelu_official(input_tensor)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(1000):
        result_pytorch = pytorch_gelu_official(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / 1000 / 1000  # 转换为秒

    print("Warming up...")
      # CUDA实现预热
    for _ in range(10):  # 预热10次
        _ = FP16_GELU.FP16_GELU8(input_tensor)

    torch.cuda.synchronize()
    print("Warmup completed.")
     # 自定义CUDA实现基准测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(1000):
        result_custom = FP16_GELU.FP16_GELU8(input_tensor)
        if result_custom is None:
            raise RuntimeError("CUDA function returned None")
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event) / 1000 / 1000  # 转换为秒

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

    # 新增带宽计算部分
    # 计算数据量（单位：GB）
    # 输入数据量 + 输出数据量（假设都是float16）
    data_size_gb = (input_tensor.numel() * 2 * 2) / 1e9  # 2 bytes per float16, 2表示输入+输出
    
    # 计算带宽（GB/s）
    pytorch_bw = data_size_gb / pytorch_time
    custom_bw = data_size_gb / custom_time
    
    # 存储带宽数据
    pytorch_bandwidth.append(pytorch_bw)
    custom_bandwidth.append(custom_bw)
    
    # 在输出中添加带宽信息
    print(f"PyTorch bandwidth: {pytorch_bw:.2f} GB/s")
    print(f"Custom CUDA bandwidth: {custom_bw:.2f} GB/s")

if __name__ == "__main__":
    # 修改后的绘图代码
    plt.figure(figsize=(12, 6))

    # 时间对比子图
    plt.subplot(1, 2, 1)
    plt.plot([size[1] for size in tensor_sizes], pytorch_times, 'o-', label='PyTorch GELU', color='blue')
    plt.plot([size[1] for size in tensor_sizes], custom_times, 'o-', label='Custom CUDA GELU', color='orange')
    plt.title('Execution Time Comparison')
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()

    # 带宽对比子图
    plt.subplot(1, 2, 2)
    plt.plot([size[1] for size in tensor_sizes], pytorch_bandwidth, 'o-', label='PyTorch', color='blue')
    plt.plot([size[1] for size in tensor_sizes], custom_bandwidth, 'o-', label='Custom CUDA', color='orange')
    plt.title('Memory Bandwidth Utilization')
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('./gelu_performance_comparison.png')
    plt.show()









