#include <torch/extension.h>
#include "ATen/ATen.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 声明模板函数（需要与.cu文件中的实现匹配）
template <int VECS_PER_THREAD, int VEC_SIZE>
void launchGEMV(float* matrix, float* vector, float* res, int rows, int cols);

// 封装函数
torch::Tensor gemv_launcher(
    torch::Tensor matrix,
    torch::Tensor vector) 
{
    CHECK_INPUT(matrix);
    CHECK_INPUT(vector);
    
    // 验证输入维度
    TORCH_CHECK(matrix.dim() == 2, "Matrix must be 2-dimensional");
    TORCH_CHECK(vector.dim() == 1, "Vector must be 1-dimensional");
    const int cols = matrix.size(1);
    TORCH_CHECK(vector.size(0) == cols, 
        "Matrix columns must match vector length");

    // 创建输出张量 [rows]
    const int rows = matrix.size(0);
    auto output = torch::empty({rows}, matrix.options());

    // 调用CUDA kernel
    if (matrix.scalar_type() == torch::kFloat32) {
        launchGEMV<4, 4>(
            matrix.data_ptr<float>(),
            vector.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            cols
        );
    } else {
        AT_ERROR("Input must be FP32 type");
    }
    
    return output;
}

// 添加从gemv.cu移动过来的包装函数
void GEMV_1_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res) {
    launchGEMV<1,4>(matrix.data_ptr<float>(), 
                   vector.data_ptr<float>(),
                   res.data_ptr<float>(),
                   matrix.size(0),
                   matrix.size(1));
}

void GEMV_2_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res) {
    launchGEMV<2,4>(matrix.data_ptr<float>(), 
                   vector.data_ptr<float>(),
                   res.data_ptr<float>(),
                   matrix.size(0),
                   matrix.size(1));
}

void GEMV_4_4(torch::Tensor matrix, torch::Tensor vector, torch::Tensor res) {
    launchGEMV<4,4>(matrix.data_ptr<float>(), 
                   vector.data_ptr<float>(),
                   res.data_ptr<float>(),
                   matrix.size(0),
                   matrix.size(1));
}

// 修改模块绑定部分
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("GEMV", &gemv_launcher, "General Matrix-Vector Multiplication");
    m.def("GEMV_1_4", &GEMV_1_4, "GEMV with VECS_PER_THREAD=1");
    m.def("GEMV_2_4", &GEMV_2_4, "GEMV with VECS_PER_THREAD=2");
    m.def("GEMV_4_4", &GEMV_4_4, "GEMV with VECS_PER_THREAD=4");
} 