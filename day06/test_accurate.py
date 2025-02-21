import torch
from torch.utils.cpp_extension import load

# 加载自定义CUDA模块
sgemm = load(name="sgemm", sources=["sgemm.cu"])

def test_correctness():
    # 测试不同尺寸的矩阵（包括非方阵）
    test_cases = [
        (256, 256, 256),  # 注意这里需要添加逗号
        # 逐步增加测试用例，先测试小尺寸
        # (128, 128, 128),
        # (256, 256, 256),
        # (512, 512, 512)
    ]
    
    # 设置相对误差容忍度（根据浮点精度调整）
    rtol = 1e-3
    atol = 1e-5
    
    for M, K, N in test_cases:
        # 生成随机矩阵（适当缩小数值范围避免溢出）
        A = torch.randn(M, K, device="cuda",dtype=torch.float32) * 0.5
        B = torch.randn(K, N, device="cuda",dtype=torch.float32) * 0.5
        
        # 计算PyTorch官方结果
        torch_result = torch.mm(A, B)
        
        # 测试所有版本
        versions = ['v0', 'v1', 'v2', 'v3']
        for ver in versions:
            try:
                # 添加同步和错误检查
                torch.cuda.synchronize()
                custom_func = getattr(sgemm, f"sgemm_fp32_{ver}")
                custom_result = custom_func(A, B)
                torch.cuda.synchronize()  # 确保内核执行完成
                
                # 验证结果一致性
                is_close = torch.allclose(custom_result, torch_result, rtol=rtol, atol=atol)
                
                # 输出结果
                status = "通过" if is_close else "失败"
                print(f"测试 {ver} 版本 [{M}x{K}] * [{K}x{N}]: {status}")
                
                # 如果失败，输出详细差异
                if not is_close:
                    diff = (custom_result - torch_result).abs().max()
                    print(f"最大绝对误差: {diff.item():.2e}")
                    print(f"平均绝对误差: {(custom_result - torch_result).abs().mean().item():.2e}")
            except Exception as e:
                print(f"执行 {ver} 版本时发生异常: {str(e)}")
                continue

if __name__ == "__main__":
    test_correctness()
