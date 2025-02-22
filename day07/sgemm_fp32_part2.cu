#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>


// 添加输入检查宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//! A矩阵按行访问,连续内存访问
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

    float* A_ptr_start = A + blockIdx.y * BM * K;
    float* B_ptr_start = B + blockIdx.x * BN;

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

        // 计算核心
        for (int k = 0; k < BK; k++) {
            // 加载A寄存器
            for (int i = 0; i < M_PER_THREAD; i++) {
                a_reg[i] = a_shared[ty * M_PER_THREAD + i][k];
            }
            
            // 加载B寄存器
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_PER_THREAD]);
            
            // 累加计算
            for (int i = 0; i < M_PER_THREAD; i++) {
                for (int j = 0; j < N_PER_THREAD; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
        }
        //* 写回
        float* C_ptr_start = C + N * blockIdx.y * BM + blockIdx.x * BN;
        for (int i = 0; i < M_PER_THREAD; i++)
            FETCH_FLOAT4(C_ptr_start[N * (ty * M_PER_THREAD + i) + tx * N_PER_THREAD + 0]) = FETCH_FLOAT4(temp[i][0]);
}




#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//! 在V6的基础上，使用寄存器索引，实现对Block中N方向float4读取
template<int BM, int BN, int BK, int M_PER_THREAD, int N_PER_THREAD, int K_PER_THREAD>
__global__ void sgemm_kernel_fp32_v8(float* A, float* B, float* C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float* A_ptr_start = A + blockIdx.y * BM * K;
    float* B_ptr_start = B + blockIdx.x * BN;

    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    float a_reg[M_PER_THREAD] = {0.0f};
    float b_reg[N_PER_THREAD] = {0.0f};
    float a_load_reg[K_PER_THREAD] = {0.0f}; //! 矩阵A转置操作使用的register
    float temp[M_PER_THREAD][N_PER_THREAD] = {0.0f};
    
    //* HBM-----> register----> shared memory
    for (int s = 0; s < K; s += BK)
    {   
        for (int i = 0; i < M_PER_THREAD; i++) //* 共享内存中一次读入16个元素，需要循环使用FETCH_FLOAT4
        {
            FETCH_FLOAT4(a_load_reg[0]) =
                 FETCH_FLOAT4(A_ptr_start[K * (ty * M_PER_THREAD + i) + s + tx * K_PER_THREAD]);
            a_shared[tx * K_PER_THREAD + 0][ty * M_PER_THREAD + i] = a_load_reg[0];
            a_shared[tx * K_PER_THREAD + 1][ty * M_PER_THREAD + i] = a_load_reg[1];
            a_shared[tx * K_PER_THREAD + 2][ty * M_PER_THREAD + i] = a_load_reg[2];
            a_shared[tx * K_PER_THREAD + 3][ty * M_PER_THREAD + i] = a_load_reg[3];
        }
        for (int i = 0; i < K_PER_THREAD; i++)
        {
            FETCH_FLOAT4(b_shared[ty * K_PER_THREAD + i][tx * N_PER_THREAD]) =
                 FETCH_FLOAT4(B_ptr_start[N * (s + ty * K_PER_THREAD + i) + tx * N_PER_THREAD]);
        }
        __syncthreads();

        // 计算核心
        for (int k = 0; k < BK; k++) {
            // 加载A寄存器
            // for (int i = 0; i < M_PER_THREAD; i++) {
            //     a_reg[i] = a_shared[ty * M_PER_THREAD + i][k];
            // }
            //! A矩阵转置后，shared memory也可以进行向量化读取
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[k][ty * N_PER_THREAD]);
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_PER_THREAD]);
            
            // 累加计算
            for (int i = 0; i < M_PER_THREAD; i++) {
                for (int j = 0; j < N_PER_THREAD; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
        }
        //* 写回
        float* C_ptr_start = C + N * blockIdx.y * BM + blockIdx.x * BN;
        for (int i = 0; i < M_PER_THREAD; i++)
            FETCH_FLOAT4(C_ptr_start[N * (ty * M_PER_THREAD + i) + tx * N_PER_THREAD + 0]) = FETCH_FLOAT4(temp[i][0]);
}


template <const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
          const int BLOCK_SIZE_N,  // width of block of A that each thread block load into shared memory
          const int BLOCK_SIZE_K,  // width of block of C that each thread block calculate
          const int THREAD_SIZE_Y, // height of block of C that each thread calculate
          const int THREAD_SIZE_X, // width of block of C that each thread calculate
          const bool ENABLE_DOUBLE_BUFFER>
__global__ void sgemm_kernel_fp32_v9(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // thread id in cur Block
    const int tid = ty * blockDim.x + tx;
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.f};
    float reg_a[THREAD_SIZE_Y] = {0.f};
    float reg_b[THREAD_SIZE_X] = {0.f};
    float ldg_a_reg[4] = {0.f};

    float *A_ptr_start = A_ptr + blockIdx.y * BLOCK_SIZE_M * K;
    float *B_ptr_start = B_ptr + blockIdx.x * BLOCK_SIZE_N;

    const int A_tile_thread_per_row = BLOCK_SIZE_K / 4; // 2
    const int B_tile_thread_per_row = BLOCK_SIZE_N / 4; // 32

    const int A_tile_tid_x = tid % A_tile_thread_per_row;
    const int A_tile_tid_y = tid / A_tile_thread_per_row;
    const int B_tile_tid_x = tid % B_tile_thread_per_row;
    const int B_tile_tid_y = tid / B_tile_thread_per_row;

    FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A_ptr_start[K * A_tile_tid_y + A_tile_tid_x * 4]);
    a_shared[0][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
    a_shared[0][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
    a_shared[0][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
    a_shared[0][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];
    FETCH_FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOAT4(B_ptr_start[N * B_tile_tid_y + B_tile_tid_x * 4]);
    __syncthreads();
    int write_stage_idx = 1;
    for (int s = BLOCK_SIZE_K; s < K; s += BLOCK_SIZE_K)
    {
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A_ptr_start[K * A_tile_tid_y + A_tile_tid_x * 4 + s]);
        a_shared[write_stage_idx][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];
        FETCH_FLOAT4(b_shared[write_stage_idx][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOAT4(B_ptr_start[N * (B_tile_tid_y + s) + B_tile_tid_x * 4]);
        write_stage_idx = write_stage_idx ^ 1;
        for (int k = 0; k < BLOCK_SIZE_K; k++)
        {
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
            FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
            FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);

            for (int i = 0; i < THREAD_SIZE_Y; i++)
                for (int j = 0; j < THREAD_SIZE_X; j++)
                    accum[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }
    write_stage_idx = write_stage_idx ^ 1;
    for (int k = 0; k < BLOCK_SIZE_K; k++)
    {
        FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
        FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);
        FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
        FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);

        for (int i = 0; i < THREAD_SIZE_Y; i++)
            for (int j = 0; j < THREAD_SIZE_X; j++)
                accum[i][j] += reg_a[i] * reg_b[j];
    }

    float *C_ptr_start = C_ptr + N * by * BLOCK_SIZE_M +
                         bx * BLOCK_SIZE_N;
    for (int i = 0; i < THREAD_SIZE_Y; i++)
    {
        FETCH_FLOAT4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X]) = FETCH_FLOAT4(accum[i][0]);
        FETCH_FLOAT4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X + 4]) = FETCH_FLOAT4(accum[i][4]);
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

// 在v7的封装函数之后添加v8的封装函数
torch::Tensor sgemm_launcher_v8(
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
    const int BM = 64;  //* BM 必须是 block.y * M_PER_THREAD
    const int BN = 64;  //* BN 必须是 block.x * N_PER_THREAD
    const int BK = 64;  //* BK 必须是 block.X * K_PER_THREAD
    const int M_PER_THREAD = 4;  // 每个线程处理4个元素
    const int N_PER_THREAD = 4;  // 每个线程处理4个元素
    const int K_PER_THREAD = 4;  // 每个线程处理4个元素
    
    dim3 block(16, 16);  // 线程块维度 (16x16=256 threads)
    dim3 grid((M + BM - 1) / BM, 
              (N + BN - 1) / BN);
    
    // 启动内核
    sgemm_kernel_fp32_v8<BM, BN, BK, M_PER_THREAD, N_PER_THREAD, K_PER_THREAD><<<grid, block>>>(
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


// 在v7的封装函数之后添加v8的封装函数
torch::Tensor sgemm_launcher_v9(
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
    
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = true;

    dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 grid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);

    // 启动内核
    sgemm_kernel_fp32_v9<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y, ENABLE_DOUBLE_BUFFER><<<grid, block>>>(
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
    m.def("sgemm_fp32_v8", &sgemm_launcher_v8, "FP32 SGEMM (Version 8)");
    m.def("sgemm_fp32_v9", &sgemm_launcher_v9, "FP32 SGEMM (Version 9)");
}
