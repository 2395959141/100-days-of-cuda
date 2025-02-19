#include <cmath>
#include <cfenv>
#include <random>
#include <bits/stdc++.h>
#include <float.h>
#include <cuda.h>
#include "cuda_runtime.h"

__device__ float gpunearbyint(float a) {
  return nearbyint(a);
}


bool CheckResult(float *out, float* groudtruth, int nums) {
    for (int i = 0; i < nums; i++) {
        if (groudtruth[i] != out[i]) {
            printf("the wrong index is %d, the groudtruth is %f, the res is %f\n", i, groudtruth[i], out[i]);
            return false;
        }
    }
    return true;
}

// py code
// def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
//     weight_max = np.max(np.abs(weight))
//     denominator = 2.0 ** (quantization_bit - 1) - 1
//     return (weight_max / denominator, 0)
template<typename T>
void GenScalePerTensorSymmetricGPU(const T* in_ptr, const int quantization_bit,
                                  const int num_elements, T* scale, T* zero_point) {
    T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
    T in_min = *std::min_element(in_ptr, in_ptr + num_elements);
    T out_max = std::max(std::abs(in_max), std::abs(in_min));
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    *scale = out_max / denominator;
    *zero_point = 0;
}


// py code
// def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
//     weight_max = np.max(weight)
//     weight_min = np.min(weight)
//     denominator = 2.0 ** quantization_bit - 1
//     scale = (weight_max - weight_min) / denominator
//     zero_point = -np.round(weight_min / scale)
//     return (scale, zero_point)

// fomula: clip(input / scale .round(), -128, 127)
template<typename T>
void QuantizationPerTensorSymmetricCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                      const int num_elements, T* out_ptr) {
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    for(int j = 0; j < num_elements; j++) {
        T out = std::nearbyint(in_ptr[j] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[j] = out;
    }
}

//! support fp32 atomicMax
inline __device__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (old != assumed);
    return __int_as_float(old);
}

//! support fp32 atomicMin
inline __device__ float atomicMin(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (old != assumed);

    return __int_as_float(old);
}

// get max and min per tensor
template<typename T>
__global__ void ReduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr,
                                     T* min_ptr, const int channel, const int HW) {
    //* dyn shared memory
    extern __shared__ unsigned char shared_max_min_memory[];
    T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
    T* shared_min = shared_max + blockDim.x;
    int total_thread_num = blockDim.x * gridDim.x;
    // reduce max and min per tensor
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;
    //* reduce max and min per tensor
    for (int i = gid; i < nums; i += total_thread_num) {
        shared_max[tid] = max(shared_max[tid], input_ptr[i]);
        shared_min[tid] = min(shared_min[tid], input_ptr[i]);
    }
    __syncthreads();
    //* intra-block compare
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < nums) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
        }
        __syncthreads();
    }
    //* atomic compare
    if (tid == 0) {
        atomicMax(max_ptr, shared_max[0]);
        atomicMin(min_ptr, shared_min[0]);
        //printf("max = %f\n", *max_ptr);
        //printf("min = %f\n", *min_ptr);
    }
}

template<typename T>
__global__ void GetScaleAndZPSymmetric(const T* max_ptr, const T* min_ptr,
                                        const int nums, const double quantization_bit,
                                        T* scale, T* zero_point) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    while (gid < nums) {
        T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
        //if (gid==0) printf("weight_max_gpu is %f, fabs(max_ptr[gid]) is %f, fabs(min_ptr[gid]) is %f\n",weight_max,fabs(max_ptr[gid]),fabs(min_ptr[gid]));
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
        scale[gid] = weight_max / denominator;
        zero_point[gid] = 0;
        gid += gridDim.x * blockDim.x;
    }
}

// get scale and zp per tensor
template<typename T>
__global__ void GetScaleAndZPAsymmetric(const T* max_ptr, const T* min_ptr, const int nums,
                                        const double quantization_bit, T* scale, T* zero_point) {
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  while (gid < nums) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    T min = -min_ptr[gid];
    T s = (max_ptr[gid] - min) / denominator;
    scale[gid] = s;
    zero_point[gid] = -1 * std::nearbyint(min / s);
    gid += gridDim.x * blockDim.x;
  }
}

// element wise operation
template<typename T>
__global__ void QuantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr,
                                          const int nums, const double quantization_bit, 
                                          T* out_ptr, const int channel, const int HW) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    T scale = *scale_ptr;
    if (gid == 0) printf("scaleGPU is %f\n", scale);
    while (gid < nums) {
        T out = gpunearbyint(in_ptr[gid] / scale);
        if (gid == 328) printf("328 in_ptr is %f, out is %f\n", in_ptr[gid], out);
        if (gid == 1587) printf("1587 in_ptr is %f, out is %f\n", in_ptr[gid], out);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;

        gid += step;
    }
}


template<typename T>
__global__ void QuantizePerTensorAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                   const int nums, const double quantization_bit, T* out_ptr,
                                  const int channel, const int HW) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;
  T scale = *scale_ptr;
  T zero_point = *zero_point_ptr;
  while (gid < nums) {

    T out = nearbyint(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}


// use macro to reduce redundant code
#define LAUNCH_GPU_KERNEL(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);


int main() {
    float milliseconds = 0;
    constexpr int nums = 400 * 20 * 10;
    constexpr int HW = 20 * 10;
    constexpr int channel = 400;
    constexpr int quantization_bit = 8;
    float* input = (float*) malloc(sizeof(float) * nums);
    float cpu_min = FLT_MAX;
    float cpu_max = FLT_MIN;
    
    for(int i = 0; i < nums; i++) {
        input[i] = -3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/6));
        cpu_min = std::min(input[i], cpu_min);
        cpu_max = std::max(input[i], cpu_max);
    }
    
    // printf("per tensor min max cpu are  %f, %f\n", cpu_min, cpu_max);
    float* output = (float*) malloc(sizeof(float) * nums);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, nums * sizeof(float));
    cudaMalloc((void **)&d_output, nums * sizeof(float));
    cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);
    // block and thread config
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxblocks, channel));
    printf("gridsize blocksize are  %d, %d\n", gridSize, blockSize);
    float *d_scale, *d_zeropoint, *d_max, *d_min;
    LAUNCH_GPU_KERNEL(ReduceMaxMinPerTensor, QuantizePerTensorSymmetric, 1, nums, HW);
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);
    // (per tensor) get CPU output to validate GPU result is right or not
    float* CPUOutput= (float*) malloc(sizeof(float) * nums);
    float* scale = (float*) malloc(sizeof(float) * 1);
    float* zeropoint = (float*) malloc(sizeof(float) * 1);
    GenScalePerTensorSymmetricGPU<float>(input, quantization_bit, nums, scale, zeropoint);
    QuantizationPerTensorSymmetricCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
    free(scale);
    free(zeropoint);

     if (CheckResult(output, CPUOutput, nums)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("first two CPUoutput are %f, %f\n", CPUOutput[0], CPUOutput[1]);
        printf("first two output are %f, %f\n", output[0], output[1]);
    }
    printf("Quantize kernel latency = %f ms\n", milliseconds);
    free(input);
    free(output);
    free(CPUOutput);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);
    cudaFree(d_zeropoint);
    cudaFree(d_max);
    cudaFree(d_min);
}