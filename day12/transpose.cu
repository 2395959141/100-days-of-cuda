#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matrix_shared_trans_v1(float* in, float* out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float s_data[32][32];

    if(row < M && col < N) {
        s_data[threadIdx.y][threadIdx.x] = in[row*N + col];
        __syncthreads();
        int n_row = blockIdx.y * blockDim.y + threadIdx.y;
        int n_col = blockIdx.x * blockDim.x + threadIdx.x;
        if (n_row < M && n_col < N) {
            out[n_row*M + n_col] = s_data[threadIdx.y][threadIdx.x];
        }
    }
}


__global__ void matrix_shared_trans_v2(float* in, float* out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float s_data[32][33];

    if(row < M && col < N) {
        s_data[threadIdx.y][threadIdx.x] = in[row*N + col];
        __syncthreads();
        int n_row = blockIdx.y * blockDim.y + threadIdx.y;
        int n_col = blockIdx.x * blockDim.x + threadIdx.x;
        if (n_row < M && n_col < N) {
            out[n_row*M + n_col] = s_data[threadIdx.y][threadIdx.x];
        }
    }
}


__global__ void matrix_shared_trans_v3(float* in, float* out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float s_data[32][32];

    if(row < M && col < N) {
        s_data[threadIdx.y][threadIdx.x] = in[row*N + col];
        __syncthreads();
        int n_row = blockIdx.y * blockDim.y + threadIdx.y;
        int n_col = blockIdx.x * blockDim.x + threadIdx.x;
        //! 从共享内存的逻辑坐标(row=x,col=y)读取数据
        //! 其映射的物理存储位置(row=x,col=x^y)
        if (n_row < M && n_col < N) {
            out[n_row*M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
        }
    }
}




