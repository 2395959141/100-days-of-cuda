import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q #! 计算没有被mask的部分
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q #! 计算对角线上的值
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN #! 计算整个序列（没有mask的情况）

    #! 指向循环的第一个位置
    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) #* 在Seq_len维度移动lo步（shape为[HEAD_DIM, SEQ_LEN]）
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0)) #* 在Seq_len维度移动lo步（shape为[HEAD_DIM, SEQ_LEN]）

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block) #! K已经转置过了

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :]) #! 所有Q索引大于K索引 都被MASK掉
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]                           #* 减去当前block的最大值
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1) #! attention score矩阵中的每行sum归一化因子

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij) #! 使用当前的最大值修改 上一个Block中的结果
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij #! 更新归一化因子

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None] #! 通过修正因子来修该前一个block的输出
        O_block = tl.dot(P_block, V_block, O_block) #! O_block += P_block @ V_block

        m_i = m_ij #! 将当前迭代的m_ij 赋值给m_i，  m_i 是上一次迭代中的最大值

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) #* [BLOCK_SIZE_KV, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) #* [HEAD_DIM, BLOCK_SIZE_KV]
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM  #! 索引到[index_batch, index_head, :, :]
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM  
    V,
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    #! 对应于Triton grid的启动维度：[ triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS ]
    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0) #* 处理的Query Block的索引

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1) #* 处理的Batch * nums_heads 索引
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS #* 在哪个Batch中
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS #* 在哪个head注意力头上

    # This allows to get the (N_CTX, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch  #! 在计算大型张量的内存偏移时，32 位整数可能不足以表示大偏移量
        + index_head.to(tl.int64) * stride_Q_head  #! 因此转为int64更加保险
    )
    #* 下面定义的每个张量， 都会定位到正确的batch 和 head维度
    #! 对于Query部分会跳过一些block, 每个程序处理一个不同的查询块
    #! 对于Key和Value部分，每个程序遍历所有的键值
    #! 创建了一个指针，指向 batch 中的 head 的正确索引
    Q_block_ptr = tl.make_block_ptr(  #* Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]  BLock的内存起始地址
        base=Q + qvk_offset,  
        shape=(SEQ_LEN, HEAD_DIM),  #* 逻辑结构
        strides=(stride_Q_seq, stride_Q_dim), #* shape中每个维度在内存中的跨步（元素数量为单位）
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), #! 定义相对于逻辑张量起始位置的偏移量
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),  #! 定义每个Block of Query的size
        order=(1, 0),
    )

    #! 指针指向一块内存块，并且设置好相邻内存块的offset
    V_block_ptr = tl.make_block_ptr( #* V[index_batch, index_head, :, :]
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), #! 表示选取一个block中的 seq_len 和 head_size的所有元素
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    
    #! 这里K需要转置，因为QK_block = Q @ K.T    体现在shape，block_shape 和 order中
    K_block_ptr = tl.make_block_ptr( #* K[index_batch, index_head, :, :]
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    #!  输出的block形状和Q的block形状一样
    O_block_ptr = tl.make_block_ptr(   #* O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    #! 前面定义好了，这里就可以通过索引来访问了
    # offs_q: the offsets for the tokens in the Q to process
    #! 对应的 head 处理的那部分 query token 索引 
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    #! 按照block_size_kv 来遍历key 和 value
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)


    #* 处理不含归一化的softmax* 所需要的参数: 每行最大值m_i 和 归一化因子l_i
    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0  #! 这里加1为了让log计算更加稳定
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    #!  BLOCK_SIZE_Q行， 完整的 HEAD_DIM 列
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    #! 循环中一个Q_block 跟所有的K_block 和 V_block 做运算。
    #! 所以要在下面的内循环前面 对 每个Block load 一次 Q
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        #! 统一处理casual 和 非 casual情况。 无论哪种情况，先计算causal那部分
        #! 如果是非 casual 情况， 则再把attention mask掉的那部分计算一遍，就得到了完整的attention输出
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block, #! 内循环中对Q的block分块，遍历KV
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q  #! 计算最大值存储地址
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))  #! 存储输出块


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape #* 每个FlashAttention内部处理 [SEQ_LEN, HEAD_DIM]维度

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), #! 在一个序列上的哪一组Query Block
            BATCH_SIZE * NUM_HEADS, #! 在哪个Batch中的哪个head注意力头上，每个Batch 都有NUM_HEADS个head
            1,
        )

        #! 反向传播需要每一行最大值 和 归一化因子，使用logsumexp技巧只需要保存一个值
        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q, 
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M, #! 为反向传播保存的信息
            O=O, #* 输出
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    #ref_O.backward(dO)
    # ref_dV, V.grad = V.grad.clone(), None  #! 保存梯度副本：即使原梯度被清空，仍然可以访问梯度值
    # ref_dK, K.grad = K.grad.clone(), None
    # ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    # tri_out.backward(dO)
    # tri_dV, V.grad = V.grad.clone(), None  #! 保存梯度副本：即使原梯度被清空，仍然可以访问梯度值
    # ref_dK, K.grad = K.grad.clone(), None
    # tri_dK, K.grad = K.grad.clone(), None
    # tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=False)
    print("PASSED")