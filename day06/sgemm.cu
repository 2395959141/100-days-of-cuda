#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>


// 添加输入检查宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//* 最基础的矩阵乘法
//! 大量读取全局内存
__global__ void sgemm_kernel_fp32_v0(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < M && y < N) {
        float temp = 0.0f;
        for (int i = 0; i < K; i++) {
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = temp;
    }
}

//* 一维线程块 + 全局内存合并 
//* 对B矩阵内存连续访问
//* 对A矩阵缓存利用
template<int BLOCK_SIZE>
__global__ void sgemm_kernel_v1(float* A, float* B, float* C, const int M, const int N, const int K)
{
    // 声明共享内存（每个block分配两个共享内存块）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 计算线程的全局坐标
    const int tx = threadIdx.x;
    const int x = blockIdx.x * BLOCK_SIZE + tx / BLOCK_SIZE;
    const int y = blockIdx.y * BLOCK_SIZE + tx % BLOCK_SIZE;

    float temp = 0.0f;

    // 分块处理K维度
    for (int tile = 0; tile < K; tile += BLOCK_SIZE) {
        // 协作加载数据到共享内存
        As[tx / BLOCK_SIZE][tx % BLOCK_SIZE] = A[x * K + tile + (tx % BLOCK_SIZE)];
        Bs[tx / BLOCK_SIZE][tx % BLOCK_SIZE] = B[(tile + (tx / BLOCK_SIZE)) * N + y];
        
        __syncthreads();  // 等待所有线程完成数据加载

        // 使用共享内存进行计算
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            temp += As[tx / BLOCK_SIZE][k] * Bs[k][tx % BLOCK_SIZE];
        }
        
        __syncthreads();  // 等待所有线程完成计算
    }

    if (x < M && y < N) {
        C[x * N + y] = temp;
    }
}

//* shared memory + sliding window
//! 计算强度不够
template<int BLOCK_SIZE>
__global__ void sgemm_kernel_v2(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    float *A_ptr_start = A + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = B + blockDim.x * blockIdx.x;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    float temp = 0.f;

    for (int s = 0; s < K; s += blockDim.x)
    {
        a_shared[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y][threadIdx.x] = B_ptr_start[threadIdx.x + (threadIdx.y + s) * N];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        __syncthreads();
    }

    C_ptr[x + y * N] = temp;
}



//* 增加每个线程工作, 提升计算强度
template<int BLOCK_SIZE, int STRIDE>
__global__ void sgemm_kernel_v3(float* A, float* B, float* C, const int M, const int N, const int K)
{
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A_ptr_start = A + STEP * ty * K;
    float *B_ptr_start = B + STEP * tx;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];
    float temp[STRIDE][STRIDE] = 0.f;

    for(int s = 0; s < K; s += STEP)
    {
        for(int i = 0; i < STRIDE; i++)
        {
            for(int j = 0; j < STRIDE; j++)
            {
                a_shared[ty + i*BLOCK_SIZE][tx + j*BLOCK_SIZE] = A_ptr_start[(ty + BLOCK_SIZE * i) * K + tx + BLOCK_SIZE * j + s];
                b_shared[ty + i*BLOCK_SIZE][tx + j*BLOCK_SIZE] = B_ptr_start[(ty + BLOCK_SIZE * i + s) * N + tx + BLOCK_SIZE * j];
            }
        }
        __syncthreads();
        for(int i = 0; i < STRIDE; i++) 
        {
            for(int j = 0; j < STRIDE; j++)
            {
                for(int k = 0; k < STEP; k++)
                {
                    temp[i][j] += a_shared[ty + i*BLOCK_SIZE][k] * b_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }

    float* C_ptr_start = C + N * ty * STEP + tx * STEP;
    for(int i = 0; i < STRIDE; i++){
        for(int j = 0; j < STRIDE; j++)
        {
            C_ptr_start[N * (ty + i*BLOCK_SIZE) + tx + j * BLOCK_SIZE] = temp[i][j];
        } 
    }
}




// 封装函数
torch::Tensor sgemm_launcher_v0(
    torch::Tensor A,
    torch::Tensor B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // 获取矩阵维度
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    // 创建输出张量
    auto C = torch::empty({M, N}, A.options());
    
    // 设置线程块和网格维度
    const int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + block.x - 1) / block.x, 
              (N + block.y - 1) / block.y);
    
    // 启动内核
    sgemm_kernel_fp32_v0<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return C;
}

// 在现有封装函数下方添加新的封装函数
torch::Tensor sgemm_launcher_v1(
    torch::Tensor A,
    torch::Tensor B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // 获取矩阵维度
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    // 创建输出张量
    auto C = torch::empty({M, N}, A.options());
    
    // 设置线程块和网格维度
    const int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE * BLOCK_SIZE); //* 使用一维线程块
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动模板化的内核
    sgemm_kernel_v1<BLOCK_SIZE><<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return C;
}

// 在现有封装函数下方添加新的封装函数
torch::Tensor sgemm_launcher_v2(
    torch::Tensor A,
    torch::Tensor B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // 获取矩阵维度
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    // 创建输出张量
    auto C = torch::empty({M, N}, A.options());
    
    // 设置线程块和网格维度
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 使用二维线程块
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动模板化的内核
    sgemm_kernel_v2<BLOCK_SIZE><<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return C;
}


// 在现有封装函数下方添加新的封装函数
torch::Tensor sgemm_launcher_v3(
    torch::Tensor A,
    torch::Tensor B) 
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // 获取矩阵维度
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    // 创建输出张量
    auto C = torch::empty({M, N}, A.options());
    
    // 设置线程块和网格维度
    const int BLOCK_SIZE = 16;
    const int STRIDE = 2;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 使用二维线程块
    dim3 grid((M + BLOCK_SIZE - 1) / (BLOCK_SIZE * STRIDE), 
              (N + BLOCK_SIZE - 1) / (BLOCK_SIZE * STRIDE));
    
    // 启动模板化的内核
    sgemm_kernel_v3<BLOCK_SIZE, STRIDE><<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return C;
}


// 绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_fp32_v0", &sgemm_launcher_v0, "FP32 SGEMM (Version 0)");
    m.def("sgemm_fp32_v1", &sgemm_launcher_v1, "FP32 SGEMM (Version 1)");
    m.def("sgemm_fp32_v2", &sgemm_launcher_v2, "FP32 SGEMM (Version 2)");
    m.def("sgemm_fp32_v3", &sgemm_launcher_v3, "FP32 SGEMM (Version 3)");
}