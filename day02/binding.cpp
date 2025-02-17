#include <torch/extension.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 声明模板函数（需要与.cu文件中的实现匹配）
template <int Vecsize>
void launchFP16GELU(half* x, half* y, int N);

// 封装函数
torch::Tensor fp16_gelu_launcher(
    torch::Tensor input) 
{
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    
    const int N = input.numel();
    const int threads = 256;
    const int blocks = (N + threads * 8 - 1) / (threads * 8);
    
    if (input.scalar_type() == torch::kHalf) {
        launchFP16GELU<8>(
            static_cast<half*>(output.data_ptr()),
            static_cast<half*>(input.data_ptr()),
            N
        );
    } else {
        AT_ERROR("Input must be FP16 type");
    }
    
    return output;
}

// 显式实例化声明（需要与.cu文件中的实例化匹配）
extern template void launchFP16GELU<8>(half*, half*, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("FP16_GELU8", &fp16_gelu_launcher, "FP16 GELU (Vecsize = 8)");
}
