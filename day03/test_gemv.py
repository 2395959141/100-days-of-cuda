import torch
from torch.utils.cpp_extension import load
import time
import matplotlib.pyplot as plt

# 加载CUDA扩展
gemv = load(
    name="gemv_ext",
    sources=['gemv_binding.cpp', 'gemv.cu'],
    verbose=True,
    extra_cuda_cflags=[
        '-O3',
        '-use_fast_math',
    ]
)

# 设置确定性计算
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)

# 测试尺寸配置（行数 x 列数）
test_shapes = [
    (1024, 1024),   # 小规模
    (2048, 2048),   # 中等规模
    (4096, 4096)    # 大规模
]

# 性能数据存储
pytorch_times = []
custom_times = []
speedups = []

def select_parameters(cols):
    param_map = {
        1024: (1, 4),
        2048: (2, 4),
        4096: (4, 4)
    }
    return param_map.get(cols, (1, 4))  # 默认使用组合A

def run_test(matrix_rows, matrix_cols, num_runs=100):
    """执行单个测试用例"""
    print(f"\nTesting shape: {matrix_rows}x{matrix_cols}")
    
    # 获取当前测试尺寸的参数组合
    vecs_per_thread, vec_size = select_parameters(matrix_cols)
    print(f"使用参数组合：VECS_PER_THREAD={vecs_per_thread}, VEC_SIZE={vec_size}")
    
    # 创建对齐的输入数据（确保内存对齐）
    matrix = torch.randn(matrix_rows, matrix_cols, device='cuda', dtype=torch.float32).contiguous()
    vector = torch.randn(matrix_cols, device='cuda', dtype=torch.float32).contiguous()
    
    # 预热阶段（消除初始化开销）
    print("Warming up...")
    warmup_output = torch.empty(matrix_rows, device='cuda', dtype=torch.float32)
    for _ in range(5):  # 预热10次
        # 预热PyTorch实现
        _ = torch.mv(matrix, vector)
        # 预热自定义实现
        if vecs_per_thread == 1:
            gemv.GEMV_1_4(matrix, vector, warmup_output)
        elif vecs_per_thread == 2:
            gemv.GEMV_2_4(matrix, vector, warmup_output)
        elif vecs_per_thread == 4:
            gemv.GEMV_4_4(matrix, vector, warmup_output)
    torch.cuda.synchronize()  # 确保所有预热操作完成
    
    # 创建正式测试用的输出张量（避免重复使用预热的内存）
    output = torch.empty(matrix_rows, device='cuda', dtype=torch.float32)
    
    # 测试PyTorch实现
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        pytorch_result = torch.mv(matrix, vector)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / num_runs
    
    # 测试自定义实现
    start_event.record()
    for _ in range(num_runs):
        if vecs_per_thread == 1:
            gemv.GEMV_1_4(matrix, vector, output)
        elif vecs_per_thread == 2:
            gemv.GEMV_2_4(matrix, vector, output)
        elif vecs_per_thread == 4:
            gemv.GEMV_4_4(matrix, vector, output)
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event) / num_runs
    
    # 验证结果正确性
    max_diff = torch.max(torch.abs(pytorch_result - output)).item()
    if not torch.allclose(pytorch_result, output, rtol=1e-4, atol=1e-4):
        print(f"❌ 结果差异过大! Max difference: {max_diff:.6f}")
    else:
        print(f"✅ 结果验证通过! Max difference: {max_diff:.6f}")
    
    # 计算加速比
    speedup = pytorch_time / custom_time
    
    print(f"PyTorch time: {pytorch_time:.6f}ms")
    print(f"Custom time: {custom_time:.6f}ms")
    print(f"speedup: {speedup:.4f}x")
    
    return pytorch_time, custom_time, speedup

# 执行所有测试用例
for rows, cols in test_shapes:
    pt_time, ct_time, speedup = run_test(rows, cols)
    pytorch_times.append(pt_time)
    custom_times.append(ct_time)
    speedups.append(speedup)

# 绘制性能对比图（仅保留执行时间对比）
plt.figure(figsize=(6, 5))

# 执行时间对比
plt.subplot(1, 1, 1)  # 修改为单图
x_labels = [f"{r}x{c}" for r,c in test_shapes]
plt.plot(x_labels, pytorch_times, 'o-', label='PyTorch')
plt.plot(x_labels, custom_times, 'o-', label='Custom CUDA')
plt.title('Execution Time Comparison')
plt.ylabel('Time (ms)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('gemv_performance.png')
plt.show() 