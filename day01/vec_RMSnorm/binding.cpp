#include <torch/extension.h>
#include "ATen/ATen.h"


// 定义检查宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// 声明 CUDA 函数（保持 extern 声明）
template <typename T>
extern void launchRMSNorm(T* input, T* scale, float eps, const int num_tokens, const int hidden_dim);

// 封装函数
torch::Tensor rms_norm_launcher(
    torch::Tensor input,
    torch::Tensor scale,
    float eps) 
{
    // 自动从input获取维度
    const int num_tokens = input.size(0);
    const int hidden_dim = input.size(1);
    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    
    // 创建输出张量
    auto output = torch::empty_like(input);
    
    if (input.scalar_type() == torch::kFloat32) {
        launchRMSNorm<float>(
            output.data_ptr<float>(),
            scale.data_ptr<float>(),
            eps,
            num_tokens,
            hidden_dim
        );
    } else {
        AT_ERROR("Unsupported data type");
    }
    
    return output;
}

// 修改后（正确）
extern template void launchRMSNorm<float>(float*, float*, float, const int, const int);
//extern template void launchRMSNorm<half>(half*, half*, float, const int, const int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_launcher, "RMS Normalization (CUDA)");
}


