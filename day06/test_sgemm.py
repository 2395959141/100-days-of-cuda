import torch
import time 
from torch.utils.cpp_extension import CUDAExtension, load

# import os
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # 根据你的GPU架构设置

sgemm = load(name="sgemm",
            sources=["sgemm.cu"],
            )

SUPPORTED_VERSIONS = ['v1', 'v2', 'v3', 'v4']
for ver in SUPPORTED_VERSIONS:
    if not hasattr(sgemm, f'sgemm_fp32_{ver}'):
        raise NotImplementedError(f"Version {ver} not implemented in CUDA code") 

def test_performance():
    # 测试不同尺寸的方阵
    sizes = [256, 512, 1024, 2048, 4096, 8192]  # 测试的矩阵尺寸
    num_runs = 100  # 每个尺寸的运行次数
    
    # 修改表头增加v3版本
   # 修改表头增加v4版本
    print(f"{'Size':<10} | {'Custom v1 (TFLOPS)':<18} | {'Custom v2 (TFLOPS)':<18} | {'Custom v3 (TFLOPS)':<18} | {'Custom v4 (TFLOPS)':<18} | {'PyTorch (TFLOPS)':<18}")
    print("-" * 120)
    
    for n in sizes:
        # 创建随机矩阵（使用CUDA设备）
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        #c = torch.zeros(n, n, device="cuda", dtype=torch.float16)
        
        #*预热CUDA缓存（增加所有需要测试的操作）
        for _ in range(3):
            sgemm.sgemm_fp32_v1(a, b)
            torch.cuda.synchronize()
            sgemm.sgemm_fp32_v2(a, b)
            torch.cuda.synchronize()
            sgemm.sgemm_fp32_v3(a, b)
            torch.cuda.synchronize()
            sgemm.sgemm_fp32_v4(a, b)
            torch.cuda.synchronize()
            # # 新增PyTorch预热
            # torch.mm(a, b)
            # torch.cuda.synchronize()

        # 新增测试自定义核函数v1
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v1(a, b)
        torch.cuda.synchronize()
        custom_v1_time = (time.time() - start) / num_runs

        # 新增测试自定义核函数v2
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v2(a, b)
        torch.cuda.synchronize()
        custom_v2_time = (time.time() - start) / num_runs

        # 新增测试自定义核函数v3
        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v3(a, b)
        torch.cuda.synchronize()
        custom_v3_time = (time.time() - start) / num_runs

        start = time.time()
        for _ in range(num_runs):
            sgemm.sgemm_fp32_v4(a, b)
        torch.cuda.synchronize()
        custom_v4_time = (time.time() - start) / num_runs

        # 测试PyTorch矩阵乘法
        start = time.time()
        for _ in range(num_runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_runs
        
        # 计算FLOPS（矩阵乘法的浮点运算次数为2*N^3）
        flops = 2 * n ** 3
        # custom_v0_tflops = (flops / 1e12) / custom_v0_time
        custom_v1_tflops = (flops / 1e12) / custom_v1_time  # 新增v1计算
        custom_v2_tflops = (flops / 1e12) / custom_v2_time  # 新增v2计算
        custom_v3_tflops = (flops / 1e12) / custom_v3_time  # 新增v3计算
        custom_v4_tflops = (flops / 1e12) / custom_v4_time  # 新增v4计算
        torch_tflops = (flops / 1e12) / torch_time
        
        # 修改输出格式包含v4结果
        print(f"{n:<10} | {custom_v1_tflops:<18.2f} | {custom_v2_tflops:<18.2f} | {custom_v3_tflops:<18.2f} | {custom_v4_tflops:<18.2f} | {torch_tflops:<18.2f}") 

if __name__ == "__main__":
    test_performance()

# sgemm = torch.utils.cpp_extension.load(name="sgemm", sources=["sgemm.cu"])

# A = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
# B = torch.randn(512, 2048, device="cuda", dtype=torch.float32)

# C = sgemm.sgemm_fp32_v0(A, B)
# print(C.shape)  # 输出: torch.Size([1024, 2048]) 

# import torch
# from torch.utils.cpp_extension import CUDAExtension, load
# sgemm = load(name="sgemm", sources=["sgemm.cu"])

# A = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
# B = torch.randn(512, 2048, device="cuda", dtype=torch.float32)

# C = sgemm.sgemm_fp32_v0(A, B)
# print(C.shape)  