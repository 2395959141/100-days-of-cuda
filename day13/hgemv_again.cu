#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

//! 参考：https://github.com/DefTruth/CUDA-Learn-Notes/blob/main/kernels/hgemv/hgemv.cu

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])

template<const int Kwarp_size = WARP_SIZE>
__device__  __forceinline__  half warp_reduce_sum_fp16 (half val)  {
    #pragma unroll
    for (int mask = Kwarp_size >> 1; mask >=1; mask >>=1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val; 
}

// HGEMV: Warp HGEMV K32
// 假设K为32的倍数，每个warp负责一行
//* grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
//* a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k32_f16_kernel(half* a, half* x, half* y, int M, int K) {
    int tx = threadIdx.x;           //0~31
    int ty = threadIdx.y;           //0~3
    int bx = blockIdx.x;            //0~M/4
    int lane = tx % WARP_SIZE;      //0~31
    int m = bx * blockDim.y + ty;   //! 起始行号 (0~M/4) * 4 + (0~3) 【块偏移 + 线程偏移】

    if (m < M) {
        int NUM_WARPS =  (K + WARP_SIZE - 1) / WARP_SIZE; //! 计算K列需要多少个warp
        half sum = __float2half(0.0f);

        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            int k = w * WARP_SIZE + lane; //! 计算当前warp的列号 【warp偏移 + 线程偏移】
            if (k < K) {
                sum = __hadd(sum, __hmul(a[m * K + k], x[k]));
            }
        }
        sum = warp_reduce_sum_fp16<WARP_SIZE>(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}


// HGEMV: Warp HGEMV K128 + half2x2
//! 假设K为128的倍数 float4
//* grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
//* a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k128_f16x4_kernel(half* a, half* x, half* y, int M, int K) {
    //! 每个线程负责4个元素, 一个warp负责128个元素
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = bx * blockDim.y + ty;

    if (m < M) {
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1 ) / 4;
        half sum = __float2half(0.0f);
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            int k = (w * WARP_SIZE + lane) * 4; //! 计算当前warp的列号 【warp偏移 + 线程偏移】
            if (k < K) {
                half2 reg_x_0 = HALF2(x[k + 0]);
                half2 reg_x_1 = HALF2(x[k + 2]);
                half2 reg_a_0 = HALF2(a[m * K + k + 0]);
                half2 reg_a_1 = HALF2(a[m * K + k + 2]);
                sum = __hadd(sum, __hadd(
                      __hadd(__hmul(reg_a_0.x, reg_x_0.x), __hmul(reg_a_0.y, reg_x_0.y)),
                      __hadd(__hmul(reg_a_1.x, reg_x_1.x), __hmul(reg_a_1.y, reg_x_1.y))
                ));
            }
        }
        sum = warp_reduce_sum_fp16<WARP_SIZE>(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}

//* --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

#define ASSERT_K_IS_MULTIBLE_OF(V) \
if (K % (V) != 0) { throw std::runtime_error("K must be multiple of "#V); }


void hgemv_k32_f16(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(32)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  hgemv_k32_f16_kernel<<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(x.data_ptr()),
    reinterpret_cast<half*>(y.data_ptr()),
    M, K
  );
}

void hgemv_k128_f16x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(128)
  
  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  hgemv_k128_f16x4_kernel<<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(x.data_ptr()),
    reinterpret_cast<half*>(y.data_ptr()),
    M, K
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hgemv_k32_f16);
  TORCH_BINDING_COMMON_EXTENSION(hgemv_k128_f16x4);
}