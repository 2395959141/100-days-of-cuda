import torch
import triton
import triton.language as tl
from triton import Config

#! 分Block量化
@triton.jit
def act_quant_kernal(
    x_ptr,
    y_ptr,
    s_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """
    修正后的量化核函数
    :param x_ptr: 输入矩阵指针
    :param y_ptr: 输出矩阵指针
    :param s_ptr: 缩放系数指针
    :param BLOCK_SIZE: 块大小（编译时常量）
    """
    pid = tl.program_id(axis=0)
    # 计算当前block的偏移量
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float16)
    
    #* 计算block的最大绝对值
    max_val = tl.max(tl.abs(x), axis=0)
    s = max_val / 448.0  #! 量化系数是 float32的
    
    # 量化操作
    y = x / s  #! s的布局是 [... , hidden_dim / block_size]
    
    # 存储结果
    tl.store(y_ptr + offset , y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    修正后的量化函数，处理维度计算问题
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0

    # 创建输出张量（保持输入维度）
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # 调整缩放因子维度为 [..., K/block_size, 1]
    s_shape = (*x.size()[:-1], x.size(-1) // block_size)
    s = x.new_empty(s_shape, dtype=torch.float16)
    
    #! 为了兼容多维度矩阵相乘
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    # 启动内核时添加显式维度参数
    act_quant_kernal[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# 修改后的自动调优配置
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=5, num_warps=8)
]
@triton.autotune(configs=fp8_gemm_configs, key=['M', 'N', 'K'])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr, b_s_ptr, 
    M, N, K,  # 修改参数顺序
    stride_am, stride_ak,  # 添加步长参数
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        stride_am (tl.constexpr): Stride for the M dimension in matrix A.
        stride_ak (tl.constexpr): Stride for the K dimension in matrix A.
        stride_bk (tl.constexpr): Stride for the K dimension in matrix B.
        stride_bn (tl.constexpr): Stride for the N dimension in matrix B.
        stride_cm (tl.constexpr): Stride for the M dimension in matrix C.
        stride_cn (tl.constexpr): Stride for the N dimension in matrix C.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K) 
    
    # 修正后的指针计算
    a_ptr = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptr = b_ptr + (offs_n[None, :] * K + offs_k[:, None]) #! 这里是按照转置之后方式取值的
     #!错误：b_ptr = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    
    # 修正缩放因子索引
    a_s_ptr = a_s_ptr + (pid_m * k)
    b_s_ptr = b_s_ptr + (pid_n * k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        # 修正mask条件
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - i * BLOCK_SIZE_K)
        mask_b = (offs_k[:, None] < K - i * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_ptr, mask=mask_b, other=0.0)
        
        a_s = tl.load(a_s_ptr + i)
        b_s = tl.load(b_s_ptr + i)
        
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        
        # 修正指针递增
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K
        #! 修正后的指针递增
        a_s_ptr += 1
        b_s_ptr += 1
    output = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr, output, mask=mask)


def fp8_gemm(a: torch.Tensor, b: torch.Tensor):
    """
    FP8矩阵乘法接口函数
    参数:
        a: (M, K) 输入矩阵，FP8格式
        b: (K, N) 输入矩阵，FP8格式
        a_scale: (M,) A矩阵的缩放因子
        b_scale: (N,) B矩阵的缩放因子
    返回:
        c: (M, N) 输出矩阵，FP32格式
    """
    # 参数校验
    #assert b.dtype == torch.float32, "输入矩阵必须是FP32，进行动态量化"
    assert a.is_contiguous() and b.is_contiguous(), "输入矩阵必须是连续内存布局"
    
    a_fp8, a_scale = act_quant(a)
    b_fp8, b_scale = act_quant(b)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )
    
    # 修改后的内核调用
    fp8_gemm_kernel[grid](
        a_fp8, b_fp8, c,
        a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),  # 添加步长参数
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # 移除 GROUP_SIZE_M 参数
    )
    return c




