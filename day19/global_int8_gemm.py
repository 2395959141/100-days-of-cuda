import torch
import triton
import triton.language as tl
from triton import Config


@triton.autotune(
    configs=[
        # 基础配置适合小尺寸矩阵
        Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            num_warps=4,
            num_stages=3,
        ),
        # 中等块尺寸配置
        Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            num_warps=8,
            num_stages=3,
        ),
        # 大块尺寸配置，适合大矩阵
        Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_warps=8,
            num_stages=4,
        ),
        # 横向扩展配置
        Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4},
            num_warps=4,
            num_stages=5,
        ),
    ],
    # 添加启发式函数来选择合适配置
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernal_int8(
    a_ptr,
    b_ptr,
    c_ptr,
    scala_a_ptr, scala_b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m % num_pid_in_group) % group_size_m)
    pid_n = (pid_m % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    #* 加载scale
    scale_a = tl.load(scala_a_ptr)
    scale_b = tl.load(scala_b_ptr)

    #* 使用int32累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # # 修正指针计算逻辑
    # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 修改后的主循环
    for k in range(0, K, BLOCK_SIZE_K):
        # mask_k = k + tl.arange(0, BLOCK_SIZE_K) < K
        # a = tl.load(a_ptrs, mask=mask_k[None, :], other=0)
        # b = tl.load(b_ptrs, mask=mask_k[:, None], other=0)

        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)

        #! 使用int矩阵乘法指令
        acc += tl.dot(a, b) 
        
    #! 反量化到float 32   
    output = acc.to(tl.float32) * (scale_a * scale_b)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn , output)


#! 全局量化策略
def matmul_int8(a: torch.Tensor, b: torch.Tensor, activation: str = " "):
    # 确保张量在GPU上
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    
    # 添加内存连续性保证
    a = a.contiguous()
    b = b.contiguous()
    
    scala_a = (a.abs().max() / 127.0).clamp(min=1e-8)
    scala_b = (b.abs().max() / 127.0).clamp(min=1e-8)

    a_int8 = (a / scala_a).round().clamp(-127, 127).to(torch.int8)
    b_int8 = (b / scala_b).round().clamp(-127, 127).to(torch.int8)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device)
    
    # 修改网格配置为动态计算
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
    
    # 简化内核调用参数
    gemm_kernal_int8[grid](
        a_int8, b_int8, c,
        scala_a, scala_b,
        M, N, K,
        a_int8.stride(0), a_int8.stride(1),
        b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
        # 完全移除num_warps和num_stages参数
    )
    return c