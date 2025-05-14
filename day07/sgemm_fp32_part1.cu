#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>


// 添加输入检查宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//! A矩阵按行访问（连续内存访问）
//! B矩阵按列访问 会导致bank conflict
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//* 分tile不一定非得是方块
template<int BM, int BN, int BK, int NUM_PER_THREAD>
__global__ void sgemm_kernel_fp32_v5(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float* A_ptr_start = A + blockIdx.y * BM * K;
    float* B_ptr_start = B + blockIdx.x * BN;

    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    float temp[NUM_PER_THREAD] = {0.0f};
    
    for(int s = 0; s < K; s += BK)
    {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[K * ty + s + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[N * (s + ty) + tx * NUM_PER_THREAD]);
        //a_shared[ty][tx * NUM_PER_THREAD + 0] = A_ptr_start[K * ty + tx * NUM_PER_THREAD];
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREAD; i++) //* 每个线程负责计算多个结果
        {
            for (int k = 0; k < BK; k++)  //* 遍历分块维度进行累加
            {
                temp[i] += a_shared[ty][k] * b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    float* C_ptr_start = C + N * blockIdx.y * BM + blockIdx.x * BN;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        C_ptr_start[ty * N + tx * NUM_PER_THREAD + i] = temp[i];
    }
}


//! 注意：这里并没有实现float4读取，因为寄存器中的数据不连续
//! shared memory中的计算内积转外积，借助寄存器实现二级缓存
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//! 这里对二维blockdim进行重排序，便于寄存器索引
template<int BM, int BN, int BK, int NUM_PER_THREAD>
__global__ void sgemm_kernel_fp32_v6(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx; //! 二维索引 ---> 一维索引
    int ctx = tid % 16; //! 对一维索引进行重排序
    int cty = tid / 16; //! 为了便于对寄存器进行索引


    float* A_ptr_start = A + blockIdx.y * BM * K;
    float* B_ptr_start = B + blockIdx.x * BN;

    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    constexpr int REG_NUM = NUM_PER_THREAD / 2; //! 一个线程求4个数，跟向量化读取一致
    float a_reg[REG_NUM] = {0.0f};
    float b_reg[REG_NUM] = {0.0f};
    float temp[REG_NUM][REG_NUM] = {0.0f};
    
    for(int s = 0; s < K; s += BK)
    {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[K * ty + s + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[N * (s + ty) + tx * NUM_PER_THREAD]);
        //a_shared[ty][tx * NUM_PER_THREAD + 0] = A_ptr_start[K * ty + tx * NUM_PER_THREAD];
        __syncthreads();

        for (int k = 0; k < BK; k++) //* 每个线程负责计算多个结果
        {
            a_reg[0] = a_shared[cty * 2][k];
            a_reg[1] = a_shared[cty * 2 + 1][k];
            b_reg[0] = b_shared[k][ctx * 2];
            b_reg[1] = b_shared[k][ctx * 2 + 1];
            for (int i = 0; i < REG_NUM; i++)
            {
                for (int j = 0; j < REG_NUM; j++)
                {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float* C_ptr_start = C + N * blockIdx.y * BM + blockIdx.x * BN;
    for (int i = 0; i < REG_NUM; i++)
    {
        for (int j = 0; j < REG_NUM; j++)
        {
            C_ptr_start[N * (cty * 2 + i) + ctx * 2 + j] = temp[i][j];
        }
    }
}



#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//! 在V6的基础上，使用寄存器索引，实现对Block中N方向float4读取
template<int BM, int BN, int BK, int M_PER_THREAD, int N_PER_THREAD, int K_PER_THREAD>
__global__ void sgemm_kernel_fp32_v7(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float* A_ptr_start = A + blockIdx.y * M_PER_THREAD * K;
    float* B_ptr_start = B + blockIdx.x * N_PER_THREAD;

    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    float a_reg[M_PER_THREAD] = {0.0f};
    float b_reg[N_PER_THREAD] = {0.0f};
    float temp[M_PER_THREAD][N_PER_THREAD] = {0.0f};
    
    //* HBM---->shared memory
    for (int s = 0; s < K; s += BK)
    {   
        for (int i = 0; i < M_PER_THREAD; i++) //* 共享内存中一次读入16个元素，需要循环使用FETCH_FLOAT4
        {
            FETCH_FLOAT4(a_shared[ty * M_PER_THREAD + i][tx * K_PER_THREAD]) =
                 FETCH_FLOAT4(A_ptr_start[K * (ty * M_PER_THREAD + i) + s + tx * K_PER_THREAD]);
        }
        for (int i = 0; i < K_PER_THREAD; i++)
        {
            FETCH_FLOAT4(b_shared[ty * K_PER_THREAD + i][tx * N_PER_THREAD]) =
                 FETCH_FLOAT4(B_ptr_start[N * (s + ty * K_PER_THREAD + i) + tx * N_PER_THREAD]);
        }
        __syncthreads();
        //* shared memory---->register
        for (int k = 0; k < K_PER_THREAD; k++)
        {
            a_reg[0] = a_shared[ty * M_PER_THREAD][k]; //! A矩阵行主序，因为A是按列取的，内存不连续，不能使用向量化读取
            a_reg[1] = a_shared[ty * M_PER_THREAD + 1][k];
            a_reg[2] = a_shared[ty * M_PER_THREAD + 2][k];
            a_reg[3] = a_shared[ty * M_PER_THREAD + 3][k];
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_PER_THREAD]); //! B矩阵行主序，而B的读取是按行来的，可以使用向量化读取
            for (int i = 0; i < M_PER_THREAD; i++)
            {
                for (int j = 0; j < N_PER_THREAD; j++)
                {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        __syncthreads();
        }
        //* 写回
        float* C_ptr_start = C + N * blockIdx.y * M_PER_THREAD + blockIdx.x * N_PER_THREAD;
        for (int i = 0; i < M_PER_THREAD; i++)
            for (int j = 0; j < N_PER_THREAD; j++)
                C_ptr_start[N * (ty * M_PER_THREAD + i) + tx * N_PER_THREAD + j] = temp[i][j];
    }
}


// 添加v5版本的封装函数
torch::Tensor sgemm_launcher_v5(
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
    
    // 设置分块参数和线程块维度
    const int BM = 32;  // 每个块处理的行数
    const int BN = 32;  // 每个块处理的列数
    const int BK = 32;  // K维度分块大小
    const int NUM_PER_THREAD = 4;  // 每个线程处理4个元素
    
    dim3 block(8 , 32);  // 线程块维度 (32, 128)
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);         // 网格Y维度
    
    // 启动内核
    sgemm_kernel_fp32_v5<BM, BN, BK, NUM_PER_THREAD><<<grid, block>>>(
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

// 添加v6版本的封装函数
torch::Tensor sgemm_launcher_v6(
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
    
    // 设置分块参数和线程块维度
    const int BM = 32;  // 每个块处理的行数
    const int BN = 32;  // 每个块处理的列数
    const int BK = 32;    // K维度分块大小
    const int NUM_PER_THREAD = 4;  // 每个线程处理4个元素
    
    dim3 block(8, 32);  // 线程块维度 (16x8=128 threads)
    dim3 grid((M + BM - 1) / BM, 
              (N + BN - 1) / BN);
    
    // 启动内核
    sgemm_kernel_fp32_v6<BM, BN, BK, NUM_PER_THREAD><<<grid, block>>>(
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


// 添加v7版本的封装函数
torch::Tensor sgemm_launcher_v7(
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
    
    // 设置分块参数和线程块维度
    const int BM = 64;  // 每个块处理的行数
    const int BN = 64;  // 每个块处理的列数
    const int BK = 64;    // K维度分块大小
    const int M_PER_THREAD = 4;  // 每个线程处理4个元素
    const int N_PER_THREAD = 4;  // 每个线程处理4个元素
    const int K_PER_THREAD = 4;  // 每个线程处理4个元素
    dim3 block(16, 16);  // 线程块维度 (16x8=128 threads)
    dim3 grid((M + BM - 1) / BM, 
              (N + BN - 1) / BN);
    
    // 启动内核
    sgemm_kernel_fp32_v7<BM, BN, BK, M_PER_THREAD, N_PER_THREAD, K_PER_THREAD><<<grid, block>>>(
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



// 更新Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_fp32_v5", &sgemm_launcher_v5, "FP32 SGEMM (Version 5)");
    m.def("sgemm_fp32_v6", &sgemm_launcher_v6, "FP32 SGEMM (Version 6)");
    m.def("sgemm_fp32_v7", &sgemm_launcher_v7, "FP32 SGEMM (Version 7)");
}
