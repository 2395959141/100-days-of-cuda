# Project Progress and Tasks 项目进展和任务

- **Bro in CUDA**: [https://github.com/a-hamdi/cuda](https://github.com/a-hamdi/cuda)
- **CUDA 中的兄弟**: [https://github.com/a-hamdi/cuda](https://github.com/a-hamdi/cuda)
- **Mentor** 🚀: [https://github.com/hkproj](https://github.com/hkproj) | [https://github.com/hkproj/100-days-of-gpu](https://github.com/hkproj/100-days-of-gpu)
- **导师**: [https://github.com/hkproj](https://github.com/hkproj) | [https://github.com/hkproj/100-days-of-gpu](https://github.com/hkproj/100-days-of-gpu)

## Mandatory and Optional Tasks 必做任务和选做任务

| Day 天 | Task Description 任务描述 |
|--------|----------------------------|
| D15    | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer). 强制 FA2-Forward: 实现 FA2 的前向传递（例如，自定义神经网络层）。 |
| D20    | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation). 强制 FA2-反向传递: 实现 FA2 的反向传递（例如，梯度计算）。 |
| D20    | **Optional Fused Chunked CE Loss + Backwards**: Fused implementation of chunked cross-entropy loss with backward pass. Can use Liger Kernel as a reference implementation. 可选的融合分块交叉熵损失 + 反向传递: 融合分块交叉熵损失的反向传递实现。可以参考 Liger Kernel 作为参考实现。 |

# Project Progress by Day

| Day   | Files & Summaries |
|-------|-------------------|
| day1  | **RMSnorm_vec.cu**: Implementation of RMS normalization using CUDA, including warp and block-level reduction for efficient computation. |
| day2  | **FP16_GELU.cu**: Implemented a vectorized FP16 GELU kernel and benchmarked it against PyTorch's implementation. |
| day3  | **gemv.cu**: Implemented a vectorized FP32 GEMV kernel and benchmarked it against PyTorch's implementation. **Issue**: Bandwidth measurement problem remains unresolved. |
| day4  | **per_tensor_quantize.cu**: Implemented per-tensor symmetric and asymmetric INT8 quantization. Results were verified on a CPU. |
| day5  | **fused_bias_mask_scale_and_add_fp16.cu**: Implemented a vectorized FP16 fused bias, mask, scale, and add kernel. |
| day6  | **optimized_kernel.cu**: |
|       | 1. v1: Added global memory coalesced access and shared memory caching |
|       | 2. v2: Implemented shared memory and sliding window technique |
|       | 3. v3: Developed strided shared memory access to improve compute intensity |
|       | 4. v4: Created vectorized reads based on v3 |
| day7  | **sgemm_kernel_fp32.cu**: |
|       | 1. v5: Implemented vectorized reads, but column-wise access of matrix B caused bank conflicts |
|       | 2. v6: Transformed inner product to outer product in shared memory, implemented secondary caching using registers, but without float4 reads |
|       | 3. v7: Based on v6, implemented float4 reads in N-direction of the block using register indexing |
|       | 4. v8: Added transpose storage of matrix A to shared memory via register buffering |
|       | 5. v9: Introduced double buffering technique using two shared memory buffers to hide memory loading latency |
| day8  | **attention_mask.cu**: |
|       | 1. 实现了基本的attention mask CUDA kernel，支持不同序列长度的mask计算 |
|       | 2. 考虑了repeat kv（key-value重复）的情况，但尚未进行充分测试 |
| day9  | **cublas_gemm.cu**: |
|       | 1. 实现了使用cublas进行矩阵乘法的基本示例 |
|       | 2. 特别注意：cublas默认使用列主序存储（column-major），与PyTorch的行主序（row-major）不同 |
|       | 3. 解决方法：在使用cublas时，可以通过转置输入矩阵来模拟行主序行为 |
|       | 4. 性能优化：使用cublasLt API进行更灵活的矩阵乘法配置 |
| day10 | **softmax_fp32.cu**: |
|       | 1. 实现了一个简洁的FP32 softmax算子 |
|       | 2. 使用了warp级别的并行归约来优化性能 |
|       | 3. 进行了数值稳定性测试，确保在输入值较大时不会出现数值溢出问题 |
|       | 4. 与PyTorch的softmax实现进行了对比测试，结果一致 |
| day11 | **concat_kv.cu**: |
|       | 1. 实现了拼接KV缓存的CUDA kernel功能 |
|       | 2. 支持高效地将新的key和value向量拼接到现有的KV缓存中 |
|       | 3. 针对不同序列长度和批量大小进行了性能优化 |
|       | 4. 通过与PyTorch实现的对比测试验证了正确性 |
| day12 | **transpose_bank_conflict.cu**: |
|       | 1. 实现了一个简单的矩阵转置算子，用于学习 shared memory 中的 bank conflict 概念 |
|       | 2. 在 NVIDIA 4060 laptop 上进行了测试，但未能复现博客中提到的 shared memory 存储过程中的 bank conflict 信息 |
|       | 3. 可能原因：现代 GPU 架构对 bank conflict 进行了优化，或者测试矩阵规模不够大 |
| day13 | **hgemv.cu**: |
|       | 1. 实现了半精度（FP16）的矩阵向量乘法（GEMV）算子 |
|       | 2. 与PyTorch的半精度GEMV实现进行了性能对比和精度验证 |
| day14 | **triton_matmul.py**: |
|       | 1. 学习了使用Triton框架实现高效矩阵乘法 |
|       | 2. 参考Triton官方文档，实现了基于分组（group）的矩阵乘法写法 |
|       | 3. 理解了Triton如何通过分组的块级并行来提升L2缓存命中从而优化了矩阵输入规模大于L2缓存情况下的数据读取速度 |



