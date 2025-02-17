#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<algorithm>

//! Create an aligned block of memory to store a fixed number of elements, providing array-style access.
template<typename T, int Size> 
struct alignas(sizeof(T) * Size) AlignedVector {
    T val[Size];
    __host__ __device__ T& operator[](int i) {return val[i];}
    __host__ __device__ const T& operator[](int i) const {return val[i];}
};


__device__ float TanhApprox(float x) {
    return tanhf(x);
}


//! Implements the computation of the GELU (Gaussian Error Linear Unit) activation function.
template<typename T>
struct GeluFunctor {
    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    __device__ T operator()(T x) const {
        const T half = static_cast<T>(0.5);
        const T one = static_cast<T>(1.0);
        const T trah_in = alpha * x * (one + beta * x * x * x);
        return half * x * (one + tanh(trah_in));
    }
};


template<>
struct GeluFunctor<half> {
    static constexpr float alpha = GeluFunctor<float>::alpha;
    static constexpr float beta = GeluFunctor<float>::beta;

    __device__ half operator()(half x) const {
        const float x_float = __half2float(x);
        const float x_cubed = x_float * x_float * x_float;
        const float tanh_in = alpha * x_float * (1.0f + beta * x_cubed);
        const float tanh_out = tanhf(tanh_in);
        
        // 分步计算避免运算符混淆
        half half_0_5 = __float2half_rn(0.5f);
        half term1 = __hmul(half_0_5, x);
        half term2 = __hadd(__float2half_rn(1.0f), __float2half_rn(tanh_out));
        return __hmul(term1, term2);
    }

    __device__ void apply2(half* y, const half* x) {
        // 使用对齐的向量化加载
        const half2 x2 = __ldg(reinterpret_cast<const half2*>(__builtin_assume_aligned(x, 4)));

        const half2 x_cubed = __hmul2(__hmul2(x2, x2), x2);
        const half2 tanh_in_part = __hadd2(__float2half2_rn(1.0f), 
                                         __hmul2(__float2half2_rn(beta), x_cubed));
        const half2 tanh_in = __hmul2(__hmul2(__float2half2_rn(alpha), x2), tanh_in_part);
        
        // 新增类型转换步骤
        const float2 tanh_in_float = __half22float2(tanh_in);
        float2 tanh_out;
        tanh_out.x = TanhApprox(tanh_in_float.x);
        tanh_out.y = TanhApprox(tanh_in_float.y);
        
        const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                                            __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
        // 使用对齐的向量化存储
        *reinterpret_cast<half2*>(__builtin_assume_aligned(y, 4)) = y2;
    }
};




//! Implements the computation of the GELU activation function for FP16 data types.
template<int Vecsize>
__global__ void FP16_GELU(half* input, half* y, int N) {
    int offset = 
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * Vecsize;
    int stride = static_cast<int>(blockDim.x * gridDim.x) * Vecsize;
    GeluFunctor<half> gelu;
    //__half y_reg[Vecsize];
    using Arrt = AlignedVector<__half, Vecsize>;

    for(; offset < N; offset += stride) {
        const __half *in = input + offset;
        for(int i = 0; i < Vecsize; i += 2) {
            gelu.apply2(y + offset + i, in + i);
        }
    }
}

template __global__ void FP16_GELU<8>(half* input, half* y, int N);

template<int Vecsize>
void launchFP16GELU(half* input, half* output, int N) {
    // 计算合适的block和grid尺寸
    const int block_size = 256;  // 每个block使用256个线程
    const int grid_size = (N + Vecsize * block_size - 1) / (Vecsize * block_size);
    
    // 调用kernel函数
    FP16_GELU<Vecsize><<<grid_size, block_size>>>(input, output, N);
}

// 显式实例化模板（与kernel的实例化保持一致）
template void launchFP16GELU<8>(half* input, half* output, int N);


