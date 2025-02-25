import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.cpp_extension import load

# 使用 JIT 方式加载 CUDA 扩展
print("正在编译 CUDA 扩展模块...")
softmax_cuda_float = load(
    name="safe_softmax_float",
    sources=["softmax.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

softmax_cuda_float4 = load(
    name="safe_softmax_float4",
    sources=["softmax.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)


print("CUDA 扩展模块编译完成！")

def test_correctness(batch_size, seq_len):
    """测试自定义 softmax 实现的正确性"""
    print(f"\n测试 batch_size={batch_size}, seq_len={seq_len} 的正确性")
    
    # 创建随机输入张量
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
    
    # 计算 PyTorch 原生 softmax
    torch_start = time.time()
    y_torch = F.softmax(x, dim=1)
    torch_time = time.time() - torch_start
    
    # 计算自定义 CUDA softmax
    cuda_start = time.time()
    y_cuda = softmax_cuda_float.safe_softmax_float(x)
    cuda_time = time.time() - cuda_start
    
    # 计算误差
    abs_diff = torch.abs(y_torch - y_cuda)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"最大绝对误差: {max_diff:.8f}")
    print(f"平均绝对误差: {mean_diff:.8f}")
    print(f"PyTorch 耗时: {torch_time * 1000:.3f} ms")
    print(f"CUDA 实现耗时: {cuda_time * 1000:.3f} ms")
    print(f"速度提升: {torch_time / cuda_time:.2f}x")
    
    # 检查是否通过测试（误差应小于阈值）
    passed = max_diff < 1e-5
    print(f"测试结果: {'通过' if passed else '失败'}")
    return passed

def test_edge_cases():
    """测试边缘情况，如极大值和极小值"""
    print("\n测试边缘情况")
    
    # 创建包含极大值的输入
    x_large = torch.ones(2, 1024, device='cuda', dtype=torch.float32) * 1000
    x_large[:, 0] = 1001  # 第一个元素稍大一些
    
    # 创建包含极小值的输入
    x_small = torch.ones(2, 1024, device='cuda', dtype=torch.float32) * -1000
    x_small[:, 0] = -999  # 第一个元素稍大一些
    
    # 测试极大值
    y_torch_large = F.softmax(x_large, dim=1)
    y_cuda_large = softmax_cuda_float.safe_softmax_float(x_large)
    max_diff_large = torch.max(torch.abs(y_torch_large - y_cuda_large)).item()
    
    # 测试极小值
    y_torch_small = F.softmax(x_small, dim=1)
    y_cuda_small = softmax_cuda_float.safe_softmax_float(x_small)
    max_diff_small = torch.max(torch.abs(y_torch_small - y_cuda_small)).item()
    
    print(f"极大值测试最大误差: {max_diff_large:.8f}")
    print(f"极小值测试最大误差: {max_diff_small:.8f}")
    
    passed = max_diff_large < 1e-5 and max_diff_small < 1e-5
    print(f"边缘情况测试结果: {'通过' if passed else '失败'}")
    return passed

def benchmark(batch_sizes, seq_lens, num_runs=50):
    """对不同大小的输入进行性能基准测试"""
    print("\n性能基准测试")
    print("=" * 90)
    print(f"{'大小(batch x seq)':>15} | {'PyTorch (ms)':>12} | {'Float实现 (ms)':>12} | {'Float4实现 (ms)':>12} | {'加速比(Float)':>12} | {'加速比(Float4)':>12}")
    print("-" * 90)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # 创建输入张量
            x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
            
            # 预热所有实现
            for _ in range(5):
                F.softmax(x, dim=1)
                softmax_cuda_float.safe_softmax_float(x)
                softmax_cuda_float4.safe_softmax_float4(x)
            
            # 计时 PyTorch
            torch_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()
                F.softmax(x, dim=1)
                torch.cuda.synchronize()
                torch_times.append(time.time() - start)
            
            # 计时 Float实现
            float_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()
                softmax_cuda_float.safe_softmax_float(x)
                torch.cuda.synchronize()
                float_times.append(time.time() - start)
            
            # 计时 Float4实现
            float4_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()
                softmax_cuda_float4.safe_softmax_float4(x)
                torch.cuda.synchronize()
                float4_times.append(time.time() - start)
            
            # 计算平均时间
            avg_torch = np.mean(torch_times) * 1000
            avg_float = np.mean(float_times) * 1000
            avg_float4 = np.mean(float4_times) * 1000
            
            # 计算加速比
            speedup_float = avg_torch / avg_float if avg_float > 0 else 0
            speedup_float4 = avg_torch / avg_float4 if avg_float4 > 0 else 0
            
            print(f"{batch_size:>6} x {seq_len:<6} | "
                  f"{avg_torch:>12.3f} | "
                  f"{avg_float:>12.3f} | "
                  f"{avg_float4:>12.3f} | "
                  f"{speedup_float:>12.2f}x | "
                  f"{speedup_float4:>12.2f}x")
            
            # # 额外打印float和float4的结果对比
            # print(f"  - Float vs Float4: Float4比Float快 {avg_float/avg_float4:.2f}x")

def benchmark_large_size(batch_size=4096, seq_len=4096, num_runs=20):
    """专门测试大尺寸输入(4096x4096)下PyTorch和float4实现的性能"""
    print(f"\n大尺寸 {batch_size}x{seq_len} 输入性能测试")
    print("=" * 60)
    
    # 创建输入张量
    print(f"创建 {batch_size}x{seq_len} 的输入张量...")
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
    
    # 预热
    print("预热中...")
    for _ in range(3):
        F.softmax(x, dim=1)
        softmax_cuda_float4.safe_softmax_float4(x)
    
    # 测试PyTorch性能
    print("测试PyTorch性能...")
    torch_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        F.softmax(x, dim=1)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        torch_times.append(elapsed)
        if (i+1) % 5 == 0:
            print(f"  PyTorch第{i+1}次运行: {elapsed*1000:.3f} ms")
    
    # 测试float4性能
    print("测试float4实现性能...")
    float4_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        softmax_cuda_float4.safe_softmax_float4(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        float4_times.append(elapsed)
        if (i+1) % 5 == 0:
            print(f"  Float4第{i+1}次运行: {elapsed*1000:.3f} ms")
    
    # 计算平均时间
    avg_torch = np.mean(torch_times) * 1000
    avg_float4 = np.mean(float4_times) * 1000
    
    # 计算加速比
    speedup = avg_torch / avg_float4 if avg_float4 > 0 else 0
    
    print("\n大尺寸性能对比结果:")
    print(f"PyTorch平均耗时: {avg_torch:.3f} ms")
    print(f"Float4平均耗时: {avg_float4:.3f} ms")
    print(f"加速比: {speedup:.2f}x")
    
    # 验证结果的正确性
    with torch.no_grad():
        y_torch = F.softmax(x, dim=1)
        y_float4 = softmax_cuda_float4.safe_softmax_float4(x)
        max_diff = torch.max(torch.abs(y_torch - y_float4)).item()
    
    print(f"最大误差: {max_diff:.8f}")
    print(f"结果{'正确' if max_diff < 1e-5 else '不正确'}")

def test_float4_implementation(batch_size, seq_len):
    """专门测试float4向量化实现的正确性和性能"""
    print(f"\n测试float4实现 batch_size={batch_size}, seq_len={seq_len}")
    
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
    
    # 预热
    for _ in range(3):
        softmax_cuda_float4.safe_softmax_float4(x)
    
    # 正确性测试
    torch_start = time.time()
    y_torch = F.softmax(x, dim=1)
    torch_time = time.time() - torch_start
    
    cuda_start = time.time()
    y_cuda = softmax_cuda_float4.safe_softmax_float4(x)
    cuda_time = time.time() - cuda_start
    
    abs_diff = torch.abs(y_torch - y_cuda)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"float4实现最大绝对误差: {max_diff:.8f}")
    print(f"float4实现平均绝对误差: {mean_diff:.8f}")
    print(f"PyTorch耗时: {torch_time * 1000:.3f} ms")
    print(f"float4实现耗时: {cuda_time * 1000:.3f} ms")
    print(f"速度提升: {torch_time / cuda_time:.2f}x")
    
    passed = max_diff < 1e-5
    print(f"float4测试结果: {'通过' if passed else '失败'}")
    return passed

def main():
    print("开始测试 CUDA Softmax 实现")
    
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("错误：CUDA 不可用。请确保安装了 CUDA 并且可以被 PyTorch 检测到。")
        return
    
    # 运行正确性测试
    test_cases = [
        (256,1024),   
        (512,1024), 
        (1024,1024),  
        (2048,1024),
        (4096,1024),
        (8192,1024),
    ]
    
    all_passed = True
    for batch_size, seq_len in test_cases:
        print(f"\n{'='*30} 测试用例 {batch_size}x{seq_len} {'='*30}")
        # 测试原始实现
        print("\n[测试原始float实现]")
        all_passed &= test_correctness(batch_size, seq_len)
        
        # 测试float4实现
        print("\n[测试float4向量化实现]")
        all_passed &= test_float4_implementation(batch_size, seq_len)
    
    # 测试边缘情况
    all_passed &= test_edge_cases()
    
    # 运行性能基准测试
    benchmark(
        batch_sizes=[256,512,1024,2048,4096,8192],
        seq_lens=[1024]
    )
    
    # 专门测试大尺寸性能
    benchmark_large_size(4096, 4096)
    
    print("\n总体测试结果:", "全部通过" if all_passed else "有失败的测试")

if __name__ == "__main__":
    main()
