import torch
from torch.utils.cpp_extension import load
import time
import os
import matplotlib.pyplot as plt

# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 设置CUDA路径
# cuda_home = "/usr/local/cuda-12.6"
# os.environ["CUDA_HOME"] = cuda_home
# os.environ["CUDA_PATH"] = cuda_home

# sources = [
#     os.path.join(current_dir, 'binding.cpp'),
#     os.path.join(current_dir, 'RMSnorm_vec.cu')
# ]
sources = [
    'binding.cpp',
    'RMSnorm_vec.cu'
]

try:
    RMSNorm = load(
        name="rms_norm_ext",
        sources=sources,
        verbose=True,
    )
    # 添加检查
    if not hasattr(RMSNorm, 'rms_norm'):
        raise RuntimeError("RMSNorm extension loaded but rms_norm function not found")
    
except Exception as e:
    print(f"编译错误: {str(e)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    #print(f"使用的CUDA路径: {cuda_home}")
    raise

print("Custom CUDA RMSNorm extension loaded successfully.")

def pytorch_rms_norm(x:torch.Tensor, g:float, eps:float=1e-5):
    """PyTorch原生实现的RMSNorm"""
    s_rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    y = x * s_rms * g
    return y

# 在测试前添加确定性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 使用固定随机种子
torch.manual_seed(42)

# 测试不同张量大小
tensor_sizes = [
    (4096, 512),
    (4096, 768),
    (4096, 1024),   # 1M元素
    (4096, 1536),
    (4096, 2048),   # 4M元素
    (4096, 4096),   # 16M元素
    #(8192, 8192),   # 64M元素
]

def benchmark_rms_norm(pytorch_steps=100, cuda_steps=100, warmup_steps=10):
    """封装后的性能测试函数"""
    # 初始化性能数据存储
    pytorch_times = []
    custom_times = []
    sizes = []
    pytorch_bandwidth = []
    custom_bandwidth = []

    for size in tensor_sizes:
        print("=" * 50)
        print(f"Testing size: {size}")
        print("=" * 50)
        
        # 生成测试数据（注意：需要与C++代码的输入参数匹配）
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        num_tokens, hidden_dim = input_tensor.shape  # 自动获取维度
        scale = torch.ones(hidden_dim, device='cuda', dtype=torch.float32)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Scale tensor shape: {scale.shape}")
        print(f"Num tokens: {num_tokens}")
        print(f"Hidden dim: {hidden_dim}")
        
        # 在测试循环中添加：
        assert scale.shape == (hidden_dim,), "Scale维度必须等于hidden_dim"
        assert input_tensor.dim() == 2, "输入必须是2D张量"
        assert scale.dim() == 1, "Scale必须是1D张量"
        
        # 在调用自定义内核前添加维度检查
        assert hidden_dim <= 8192, "hidden_dim exceeds maximum supported size"
        assert num_tokens <= 65535, "num_tokens exceeds maximum grid dimension"
        
        # ========== 添加warmup步骤 ==========
        # PyTorch实现warmup
        print("Warming up PyTorch...")
        for _ in range(warmup_steps):
            _ = pytorch_rms_norm(input_tensor, scale, 1e-5)
        torch.cuda.synchronize()

        # 自定义CUDA实现warmup
        print("Warming up CUDA...")
        for _ in range(warmup_steps):
            _ = RMSNorm.rms_norm(input_tensor, scale, 1e-5)
        torch.cuda.synchronize()
        # ========== warmup结束 ==========

        # PyTorch实现基准测试
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(pytorch_steps):  # 使用参数控制循环次数
            result_pytorch = pytorch_rms_norm(input_tensor, scale, 1e-5)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / pytorch_steps  # 根据实际步数计算平均时间
        
        # 自定义CUDA实现基准测试
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(cuda_steps):  # 使用参数控制循环次数
            result_custom = RMSNorm.rms_norm(input_tensor, scale, 1e-5)
        torch.cuda.synchronize()
        custom_time = (time.time() - start) / cuda_steps  # 根据实际步数计算平均时间
        
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

        # 计算数据量（单位：GB）
        # 输入数据量 + 输出数据量（假设都是float32）
        data_size_gb = (input_tensor.numel() * 4 * 2) / 1e9  # 4 bytes per float32, 2表示输入+输出
        
        # 计算带宽（GB/s）
        pytorch_bw = data_size_gb / pytorch_time
        custom_bw = data_size_gb / custom_time
        
        # 存储带宽数据
        pytorch_bandwidth.append(pytorch_bw)
        custom_bandwidth.append(custom_bw)
        
        # 在输出中添加带宽信息
        print(f"PyTorch bandwidth: {pytorch_bw:.2f} GB/s")
        print(f"Custom CUDA bandwidth: {custom_bw:.2f} GB/s")

    return pytorch_times, custom_times, sizes, pytorch_bandwidth, custom_bandwidth

# 测试代码
def test_rms_norm():
    input_tensor = torch.randn(10, 10, device='cuda')
    scale_tensor = torch.randn(10, device='cuda')
    output = RMSNorm.rms_norm(input_tensor, scale_tensor, 1e-5)
    print("Test passed!")

if __name__ == "__main__":
    # 现在可以通过参数控制测试步数
    pytorch_times, custom_times, sizes, pytorch_bw, custom_bw = benchmark_rms_norm(
        pytorch_steps=100,
        cuda_steps=100,
        warmup_steps=10
    )
    
    # 在代码末尾添加以下绘图代码
    plt.figure(figsize=(12, 6))

    # 时间对比子图
    plt.subplot(1, 2, 1)
    plt.plot([size[1] for size in tensor_sizes], pytorch_times, 'o-', label='PyTorch RMSNorm', color='blue')
    plt.plot([size[1] for size in tensor_sizes], custom_times, 'o-', label='Custom CUDA RMSNorm', color='orange')
    plt.title('Execution Time Comparison')
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()

    # 带宽对比子图
    plt.subplot(1, 2, 2)
    plt.plot([size[1] for size in tensor_sizes], pytorch_bw, 'o-', label='PyTorch', color='blue')
    plt.plot([size[1] for size in tensor_sizes], custom_bw, 'o-', label='Custom CUDA', color='orange')
    plt.title('Memory Bandwidth Utilization')
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('./rmsnorm_performance_comparison.png')
    plt.show()
