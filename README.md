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