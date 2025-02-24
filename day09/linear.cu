#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cublas_utils.cc"

//!  q*k
//*  A : q.shape=[bs, head num,  seq_len  head_size]
//*  B : k.shape=[bs, head num,  seq_len  head_size]   with trans_b = true
//*  C : output.shape=[bs, head num,  seq_len  seq_len]  
//! qk*v
//*  A : qk.shape=[bs, head num,  seq_len  seq_len]
//*  B : v.shape=[bs, head num,  seq_len  head_size]    with trans_b = false
//*  C : output.shape=[bs, head num,  seq_len  head_size]
// 修改后的linear_gemm函数
torch::Tensor linear_gemm(
    torch::Tensor input_tensor,
    torch::Tensor weight_tensor,
    bool transpose_a,
    bool transpose_b) 
{
    // 输入检查（保持与launchLinearGemm一致）
    TORCH_CHECK(input_tensor.dim() == 2 || input_tensor.dim() == 3, "Input tensor must be 2D or 3D");
    TORCH_CHECK(weight_tensor.dim() == 2, "Weight tensor must be 2D");

    // 获取CUDA句柄
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasWrapper cublas_wrapper(handle, nullptr);

    // 维度处理（完全对齐launchLinearGemm逻辑）
    int Ak = transpose_a ? weight_tensor.size(1) : weight_tensor.size(0);
    int Am = transpose_a ? weight_tensor.size(0) : weight_tensor.size(1);
    int Bk = input_tensor.dim() == 3 ? 
             input_tensor.size(2) :  // 正确取最后一维
             input_tensor.size(1);
    
    // 维度校验（对应LLM_CHECK_WITH_INFO）
    if (!transpose_a && !transpose_b) {
        TORCH_CHECK(Ak == Bk, "Weight dimension 0 must match input dimension 1");
    }

    // 转置设置（与launchLinearGemm完全一致）
    cublasOperation_t transA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // 计算输出维度（处理3D情况）
    std::vector<int64_t> output_shape;
    if (input_tensor.dim() == 3) {
        output_shape = {input_tensor.size(0), 
                       input_tensor.size(1), 
                       transpose_b ? Ak : Am};  // 对齐launchLinearGemm中的Cm计算
    } else {
        output_shape = {input_tensor.size(0), 
                       transpose_b ? Ak : Am};  // 2D情况处理
    }
    auto output = torch::empty(output_shape, input_tensor.options());

    // 设置GEMM参数（完全对齐launchLinearGemm逻辑）
    int M = Am;
    int N = input_tensor.dim() == 3 ? input_tensor.size(0) * input_tensor.size(1) : input_tensor.size(0);
    int K = Bk;                     // 输入的有效维度
    
    // Leading dimensions设置
    int lda = Am;  // 对应launchLinearGemm中的lda
    int ldb = Bk;  // 对应launchLinearGemm中的ldb
    int ldc = output_shape.back();  // 对应launchLinearGemm中的ldc

    // 调用GEMM（参数顺序与launchLinearGemm完全一致）
    cublas_wrapper.Gemm(
        transA,
        transB,
        M,      // 正确M参数
        N,      // 正确N参数
        K,      // 正确K参数
        weight_tensor.data_ptr<float>(),
        lda,
        input_tensor.data_ptr<float>(),
        ldb,
        output.data_ptr<float>(),
        ldc,
        1.0f,
        0.0f
    );
    
    return output;
}

// PyBind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_gemm", &linear_gemm, "CUDA加速的线性层GEMM计算",
          py::arg("input"),
          py::arg("weight"),
          py::arg("transpose_a") = false,
          py::arg("transpose_b") = false);
}