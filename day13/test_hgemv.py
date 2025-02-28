import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.cpp_extension import load

# 编译并加载自定义 CUDA 扩展
hgemv_cuda = load(
    name="hgemv",
    sources=["hgemv_again.cu"],
    verbose=True
)

def compare_results(a: torch.Tensor, b: torch.Tensor, name: str) -> bool:
    """比较两个张量的结果，计算相对误差"""
    if not torch.allclose(a, b, rtol=1e-3, atol=1e-3):
        abs_diff = torch.abs(a - b)
        max_diff = torch.max(abs_diff).item()
        rel_diff = torch.norm(abs_diff) / torch.norm(b)
        print(f"{name} 结果不匹配! 最大绝对误差: {max_diff}, 相对误差: {rel_diff}")
        return False
    return True

def benchmark_k32(M: int, K: int, num_iters: int = 100) -> Tuple[float, float, float]:
    """测试 K32 版本的 HGEMV"""
    # 确保 K 是 32 的倍数
    K = (K // 32) * 32
    
    # 创建输入数据
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    x = torch.randn(K, 1, dtype=torch.float16, device="cuda")
    y_custom = torch.zeros(M, 1, dtype=torch.float16, device="cuda")
    
    # PyTorch 基准测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        y_torch = torch.matmul(a, x)
        torch.cuda.synchronize()
    end = time.time()
    pytorch_time = (end - start) / num_iters
    
    # 自定义 kernel 测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        hgemv_cuda.hgemv_k32_f16(a, x, y_custom)
        torch.cuda.synchronize()
    end = time.time()
    custom_time = (end - start) / num_iters
    
    # 验证结果
    is_correct = compare_results(y_custom, y_torch, "K32")
    
    # 计算带宽 (GB/s) - 更合理的估计
    # 假设向量x可以完全缓存
    bytes_processed = (M * K * 2) + (M * 2)  # 只考虑矩阵a的读取和向量y的写入
    custom_bandwidth = bytes_processed / custom_time / 1e9
    pytorch_bandwidth = bytes_processed / pytorch_time / 1e9
    
    # 也可以保留原来的算术带宽计算以供对比
    theoretical_bytes = (M * K * 2) + (K * 2) + (M * 2)
    theoretical_custom_bw = theoretical_bytes / custom_time / 1e9
    theoretical_pytorch_bw = theoretical_bytes / pytorch_time / 1e9
    
    # 返回更多信息以便分析
    return custom_time, pytorch_time, custom_bandwidth, pytorch_bandwidth, theoretical_custom_bw, theoretical_pytorch_bw, is_correct

def benchmark_k128(M: int, K: int, num_iters: int = 100) -> Tuple[float, float, float]:
    """测试 K128 版本的 HGEMV"""
    # 确保 K 是 128 的倍数
    K = (K // 128) * 128
    
    # 创建输入数据
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    x = torch.randn(K, 1, dtype=torch.float16, device="cuda")
    y_custom = torch.zeros(M, 1, dtype=torch.float16, device="cuda")
    
    # PyTorch 基准测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        y_torch = torch.matmul(a, x)
        torch.cuda.synchronize()
    end = time.time()
    pytorch_time = (end - start) / num_iters
    
    # 自定义 kernel 测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        hgemv_cuda.hgemv_k128_f16x4(a, x, y_custom)
        torch.cuda.synchronize()
    end = time.time()
    custom_time = (end - start) / num_iters
    
    # 验证结果
    is_correct = compare_results(y_custom, y_torch, "K128")
    
    # 计算带宽 (GB/s) - 更合理的估计
    # 假设向量x可以完全缓存
    bytes_processed = (M * K * 2) + (M * 2)  # 只考虑矩阵a的读取和向量y的写入
    custom_bandwidth = bytes_processed / custom_time / 1e9
    pytorch_bandwidth = bytes_processed / pytorch_time / 1e9
    
    # 也可以保留原来的算术带宽计算以供对比
    theoretical_bytes = (M * K * 2) + (K * 2) + (M * 2)
    theoretical_custom_bw = theoretical_bytes / custom_time / 1e9
    theoretical_pytorch_bw = theoretical_bytes / pytorch_time / 1e9
    
    # 返回更多信息以便分析
    return custom_time, pytorch_time, custom_bandwidth, pytorch_bandwidth, theoretical_custom_bw, theoretical_pytorch_bw, is_correct

def run_benchmarks():
    """运行一系列不同大小的基准测试"""
    print("=" * 80)
    print(f"{'M':>10}{'K':>10}{'算子':>15}{'时间(ms)':>15}{'带宽(GB/s)':>15}{'加速比':>15}{'理论带宽(GB/s)':>20}")
    print("-" * 100)
    
    # 测试不同的矩阵大小
    test_sizes = [
        (1024, 1024),
        (2048, 1024),
        (4096, 1024),
        (8192, 1024),
        # (1024, 2048),
        # (1024, 4096),
    ]
    
    results = []
    
    for M, K in test_sizes:
        # K32 测试
        k32_time, pytorch_time, k32_bw, pytorch_bw, theoretical_k32_bw, theoretical_pytorch_bw, is_correct = benchmark_k32(M, K)
        speedup = pytorch_time / k32_time
        results.append((M, K, "K32", k32_time, pytorch_time, k32_bw, pytorch_bw, speedup))
        
        print(f"{M:>10}{K:>10}{'PyTorch':>15}{pytorch_time*1000:>15.3f}{pytorch_bw:>15.3f}{1.0:>15.3f}{theoretical_pytorch_bw:>20.3f}")
        print(f"{M:>10}{K:>10}{'HGEMV-K32':>15}{k32_time*1000:>15.3f}{k32_bw:>15.3f}{speedup:>15.3f}{theoretical_k32_bw:>20.3f}")
        
        # K128 测试 (如果 K 是 128 的倍数)
        if K % 128 == 0:
            k128_time, pytorch_time, k128_bw, pytorch_bw, theoretical_k128_bw, theoretical_pytorch_bw, is_correct = benchmark_k128(M, K)
            speedup = pytorch_time / k128_time
            results.append((M, K, "K128", k128_time, pytorch_time, k128_bw, pytorch_bw, speedup))
            print(f"{M:>10}{K:>10}{'HGEMV-K128':>15}{k128_time*1000:>15.3f}{k128_bw:>15.3f}{speedup:>15.3f}{theoretical_k128_bw:>20.3f}")
        
        print("-" * 100)
    
    # # 绘制结果图表
    # plot_results(results)

def plot_results(results):
    """绘制基准测试结果"""
    plt.figure(figsize=(15, 10))
    
    # 执行时间对比
    plt.subplot(2, 1, 1)
    for M, K, kernel, custom_time, pytorch_time, _, _, _ in results:
        label = f"{kernel} (M={M}, K={K})"
        plt.bar(label, custom_time * 1000, alpha=0.7, label=label)
        plt.bar(f"PyTorch (M={M}, K={K})", pytorch_time * 1000, alpha=0.7)
    
    plt.ylabel('执行时间 (ms)')
    plt.title('HGEMV 执行时间对比')
    plt.xticks(rotation=45, ha='right')
    
    # 带宽对比
    plt.subplot(2, 1, 2)
    for M, K, kernel, _, _, custom_bw, pytorch_bw, _ in results:
        label = f"{kernel} (M={M}, K={K})"
        plt.bar(label, custom_bw, alpha=0.7, label=label)
        plt.bar(f"PyTorch (M={M}, K={K})", pytorch_bw, alpha=0.7)
    
    plt.ylabel('带宽 (GB/s)')
    plt.title('HGEMV 带宽对比')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('hgemv_benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("错误：CUDA 不可用，请确保已安装 CUDA 并配置正确")
        exit(1)
        
    # 打印 CUDA 设备信息
    device = torch.cuda.current_device()
    print(f"使用 CUDA 设备: {torch.cuda.get_device_name(device)}")
    print(f"CUDA 计算能力: {torch.cuda.get_device_capability(device)}")
    
    # 运行基准测试
    run_benchmarks() 