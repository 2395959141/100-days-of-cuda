#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <float.h>

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)[0]))

template<const int nwarpsize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int nwarpsize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_fp32(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = fmax(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

//* 一个block中的线程数量 NUM_THREAD = 256
template<const int NUM_THREAD = 256>
__device__ float block_reduce_sum_fp32(float val) {
    constexpr int NUM_WARP = (NUM_THREAD + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARP];

    val = warp_reduce_sum_fp32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARP) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_fp32<WARP_SIZE>(val);
    val = __shfl_sync(0xffffffff, val, 0, 32);
    return val;
}

//* 一个block中的线程数量 NUM_THREAD = 256
template<const int NUM_THREAD = 256>
__device__ float block_reduce_max_fp32(float val) {
    constexpr int NUM_WARP = (NUM_THREAD + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARP];

    val = warp_reduce_max_fp32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val= (lane < NUM_WARP) ? shared[lane] : -FLT_MAX;
    val = warp_reduce_max_fp32<WARP_SIZE>(val);
    val = __shfl_sync(0xffffffff, val, 0, 32);
    return val;
}


//* 每个 block 处理一个完整的 token
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? x[idx] : -FLT_MAX;
    float max_val = block_reduce_max_fp32<NUM_THREADS>(val);
    float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
    float exp_sum = block_reduce_sum_fp32<NUM_THREADS>(exp_val);

    if(idx < N) y[idx] = exp_val / exp_sum;
}


template<const int NUM_THREADS = 256/4>
__global__ void safe_softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 4;

    // 加载4个连续元素
    float4 reg_x;
    if (idx < N) {
        reg_x = *reinterpret_cast<float4*>(x + idx);
    } else {
        reg_x.x = reg_x.y = reg_x.z = reg_x.w = -FLT_MAX;
    }
    
    reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
    reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
    reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
    reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;

    // 计算局部最大值
    float val = reg_x.x;
    val = fmaxf(val, reg_x.y);
    val = fmaxf(val, reg_x.z);
    val = fmaxf(val, reg_x.w);
    float max_val = block_reduce_max_fp32<NUM_THREADS>(val);

    // 计算指数值
    float4 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
    reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
    reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;

    // 规约求和
    float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
    float exp_sum = block_reduce_sum_fp32<NUM_THREADS>(exp_val);

    // 写入结果
    if (idx + 3 < N) {
        float4 reg_y;
        reg_y.x = reg_exp.x / exp_sum;
        reg_y.y = reg_exp.y / exp_sum;
        reg_y.z = reg_exp.z / exp_sum;
        reg_y.w = reg_exp.w / exp_sum;
        *reinterpret_cast<float4*>(y + idx) = reg_y;
    }
}


//!宏定义kernel launch , 处理不同输入的情况
#define LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H) \
safe_softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),        \
      reinterpret_cast<float*>(y.data_ptr()),        \
      N); \


#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                   \
  dim3 grid((S));                                    \
  switch ((H)) {                                     \
    case 32:                                         \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)       \
      break;                                         \
    case 64:                                         \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)       \
      break;                                         \
    case 128:                                        \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)      \
      break;                                         \
    case 256:                                        \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)      \
      break;                                         \
    case 512:                                        \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)      \
      break;                                         \
    case 1024:                                       \
      LAUNCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)     \
      break;                                         \
    default:                                         \
      throw std::runtime_error(                      \
        "only support H: 64/128/256/512/1024");        \
      break;                                         \
  } 


#define LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)            \
        safe_softmax_f32x4_per_token_kernel<(H/4)>              \
        <<<grid, block>>>(                                              \
            reinterpret_cast<float*>(x.data_ptr()),                     \
            reinterpret_cast<float*>(y.data_ptr()),                     \
            N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H) \
    dim3 block((H/4));                                            \
    dim3 grid((S));                                               \
    switch ((H)) {                                                \
        case 128:                                                 \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)\
            break;                                                \
        case 256:                                                 \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)\
            break;                                                \
        case 512:                                                 \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)\
            break;                                                \
        case 1024:                                                \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)\
            break;                                                \
        case 2048:                                                \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)\
            break;                                                \
        case 4096:                                                \
            LAUNCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)\
            break;                                                \
        default:                                                \
            throw std::runtime_error(                             \
                "only support H: 128/256/.../4096;");                \
            break;                                                \
    }


//! torch extension 接口
torch::Tensor safe_softmax_f32_per_token(torch::Tensor x) {
    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    auto y = torch::empty_like(x);  // 自动创建输出张量
    DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
    return y;
}


torch::Tensor safe_softmax_f32x4_per_token(torch::Tensor x) {
    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    auto y = torch::empty_like(x);  // 自动创建输出张量
    DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
    return y;
}


//! 使用pybind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("safe_softmax_float", &safe_softmax_f32_per_token, "安全的softmax实现（CUDA加速）",
          py::arg("input"));
    m.def("safe_softmax_float4", &safe_softmax_f32x4_per_token, "安全的softmax实现（CUDA加速）",
          py::arg("input"));
}


