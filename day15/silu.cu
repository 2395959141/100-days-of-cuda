#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
// #include "vec_utils.h"
#include <torch/extension.h>
#include "../day01/vec_RMSnorm/vec_utils.h"

//*  x * sigmoid(x)
template <typename T>
__device__ __forceinline__ T silu_forward(const T& x) {
    return (((float) x) / (1.0f + expf(-(float)x)));
}

template<>
__device__ __forceinline__ half2 silu_forward<half2>(const half2& input) {
   return make_half2(
       __float2half(silu_forward<float>(__half2float(input.x))), 
       __float2half(silu_forward<float>(__half2float(input.y)))
   );
}

//! x,y为 swiglu中是对输入的线性变换后的值。从而对得到的中间结果进行Slu和乘法
template<typename T>
__global__ void silu_and_mul_kernel(
    T* out,         //! [bs, intermedia size]
    const T* input, //! [bs, 2, intermedia size]
    const int intermedia_size) {
    const int batch_idx = blockIdx.x;
    for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
       const T x = input[batch_idx * intermedia_size * 2 + idx];
       const T y = input[batch_idx * intermedia_size * 2 + intermedia_size + idx];
       out[batch_idx * intermedia_size + idx] = silu_forward(x) * y;
    }
}

template<>
__global__ void silu_and_mul_kernel<half>(
    half* out,
    const half* input,
    const int intermedia_size) {
    const int batch_idx = blockIdx.x;
    int vec_size = Vec<half>::size;  //! Vec<half> 是half2的向量类型
    using Vec_t = typename Vec<half>::Type;
    for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x * vec_size) {
        const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * intermedia_size * 2 + idx]));
        const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * intermedia_size * 2 + intermedia_size + idx]));
        *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu_forward<Vec_t>(x), y);
    }
}
   
// template<typename T>
// void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
//     int batch_size = input->shape[0];
//     LLM_CHECK(input->shape[1] == 2);
//     int intermedia_size = input->shape[2];
//     dim3 grid(batch_size);
//     dim3 block(256);
//     silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
// }

// 封装函数，处理不同数据类型
torch::Tensor silu_and_mul_cuda(torch::Tensor input) {
    // 检查输入张量的维度
    TORCH_CHECK(input.dim() == 3, "输入张量必须是3维的 [batch_size, 2, intermedia_size]");
    TORCH_CHECK(input.size(1) == 2, "输入张量的第二个维度必须为2");
    
    // 获取输入形状
    int batch_size = input.size(0);
    int intermedia_size = input.size(2);
    
    // 创建输出张量
    auto output = torch::empty({batch_size, intermedia_size}, input.options());
    
    // 确保输入和输出张量在同一设备上
    TORCH_CHECK(input.device().is_cuda(), "输入张量必须在CUDA设备上");
    
    // 计算grid和block大小
    dim3 grid(batch_size);
    dim3 block(256);
    
    // 根据数据类型调用不同的CUDA核函数
    //! PyTorch 中的一个宏，用于根据输入张量的数据类型选择合适的CUDA核函数
    //* 第一个参数 input.scalar_type() 获取输入张量的数据类型
    //* 第二个参数 "silu_and_mul_cuda" 是函数名，用于错误报告
    //* 第三个参数是一个lambda函数，包含根据数据类型执行的具体代码
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "silu_and_mul_cuda", ([&] {
        if (input.scalar_type() == at::ScalarType::Half) {
            // 使用half数据类型
            silu_and_mul_kernel<half><<<grid, block>>>(
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                intermedia_size
            );
        } else {
            // 使用float或double数据类型
            silu_and_mul_kernel<scalar_t><<<grid, block>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                intermedia_size
            );
        }
    }));
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string errorMsg = "CUDA错误: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(errorMsg);
    }
    
    return output;
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul", &silu_and_mul_cuda, 
          "SiLU激活函数和乘法操作 (CUDA)",
          py::arg("input"));
} 