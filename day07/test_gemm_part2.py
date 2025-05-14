import torch
import time 
from torch.utils.cpp_extension import CUDAExtension, load

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # 根据你的GPU架构设置

sgemm = load(name="sgemm",
            sources=["sgemm_fp32_part2.cu"],
            )

SUPPORTED_VERSIONS = ['v5','v6','v7', 'v8', 'v9']
for ver in SUPPORTED_VERSIONS:
    if not hasattr(sgemm, f'sgemm_fp32_{ver}'):
        raise NotImplementedError(f"Version {ver} not implemented in CUDA code") 

def test_performance():
    sizes = [256, 512, 1024, 2048, 4096, 8192]  # 测试的矩阵尺寸
    num_runs = 200  # 每个尺寸的运行次数
    
    print(f"{'Size':<10} | {'Custom v5 (TFLOPS)':<18} | {'Custom v6 (TFLOPS)':<18} | {'Custom v7 (TFLOPS)':<18} | {'Custom v8 (TFLOPS)':<18} | {'Custom v9 (TFLOPS)':<18} | {'PyTorch (TFLOPS)':<18}")  # 添加v8列
    print("-" * 118)  # 调整分隔线长度
    
    for n in sizes:
        # 创建随机矩阵（使用CUDA设备）
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        
        #*预热CUDA缓存（增加所有需要测试的操作）
        for _ in range(10):
            sgemm.sgemm_fp32_v5(a, b)
            sgemm.sgemm_fp32_v6(a, b)
            sgemm.sgemm_fp32_v7(a, b)
            sgemm.sgemm_fp32_v8(a, b)  
            sgemm.sgemm_fp32_v9(a, b)  
            torch.cuda.synchronize()

        # 测试自定义核函数v5
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v5(a, b)
        torch.cuda.synchronize()
        custom_v5_time = (time.time() - start) / num_runs

        # 测试自定义核函数v6
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v6(a, b)
        torch.cuda.synchronize()
        custom_v6_time = (time.time() - start) / num_runs

        # 测试自定义核函数v7
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v7(a, b)
        torch.cuda.synchronize()
        custom_v7_time = (time.time() - start) / num_runs

        # 新增v8的测试代码
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v8(a, b)
        torch.cuda.synchronize()
        custom_v8_time = (time.time() - start) / num_runs

        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v9(a, b)
        torch.cuda.synchronize()
        custom_v9_time = (time.time() - start) / num_runs

        # 测试PyTorch矩阵乘法
        start = time.time()
        for _ in range(num_runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_runs
        
        # 计算FLOPS（矩阵乘法的浮点运算次数为2*N^3）
        flops = 2 * n ** 3
        custom_v5_tflops = (flops / 1e12) / custom_v5_time
        custom_v6_tflops = (flops / 1e12) / custom_v6_time
        custom_v7_tflops = (flops / 1e12) / custom_v7_time
        custom_v8_tflops = (flops / 1e12) / custom_v8_time  
        custom_v9_tflops = (flops / 1e12) / custom_v9_time
        torch_tflops = (flops / 1e12) / torch_time
        
        # 修改输出格式包含v8
        print(f"{n:<10} | {custom_v5_tflops:<18.2f} | {custom_v6_tflops:<18.2f} | {custom_v7_tflops:<18.2f} | {custom_v8_tflops:<18.2f} | {custom_v9_tflops:<18.2f} | {torch_tflops:<18.2f}") 

if __name__ == "__main__":
    try:
        test_performance()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        raise