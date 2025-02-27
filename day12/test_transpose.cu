#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "transpose.cu"

// 测试函数
void test_bandwidth(int M, int N) {
    // 计算矩阵大小
    size_t size = M * N * sizeof(float);
    
    // 分配主机内存
    float* h_in = (float*)malloc(size);
    float* h_out_v1 = (float*)malloc(size);
    float* h_out_v2 = (float*)malloc(size);
    float* h_out_v3 = (float*)malloc(size);
    
    // 初始化输入矩阵
    for (int i = 0; i < M * N; i++) {
        h_in[i] = (float)i;
    }
    
    // 分配设备内存
    float *d_in, *d_out_v1, *d_out_v2, *d_out_v3;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out_v1, size);
    cudaMalloc(&d_out_v2, size);
    cudaMalloc(&d_out_v3, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // 设置线程块和网格大小
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    
    // 测试v1版本
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        matrix_shared_trans_v1<<<gridDim, blockDim>>>(d_in, d_out_v1, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_v1 = 0;
    cudaEventElapsedTime(&milliseconds_v1, start, stop);
    
    // 测试v2版本
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        matrix_shared_trans_v2<<<gridDim, blockDim>>>(d_in, d_out_v2, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_v2 = 0;
    cudaEventElapsedTime(&milliseconds_v2, start, stop);
    
    // 测试v3版本
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        matrix_shared_trans_v3<<<gridDim, blockDim>>>(d_in, d_out_v3, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_v3 = 0;
    cudaEventElapsedTime(&milliseconds_v3, start, stop);
    
    // 计算带宽
    double total_bytes = 2.0 * M * N * sizeof(float) * 100;
    double bandwidth_v1 = (total_bytes / (milliseconds_v1 / 1000.0)) / 1e9;
    double bandwidth_v2 = (total_bytes / (milliseconds_v2 / 1000.0)) / 1e9;
    double bandwidth_v3 = (total_bytes / (milliseconds_v3 / 1000.0)) / 1e9;
    
    // 输出结果
    std::cout << "矩阵大小: " << M << "x" << N << std::endl;
    std::cout << "v1带宽: " << bandwidth_v1 << " GB/s" << std::endl;
    std::cout << "v2带宽: " << bandwidth_v2 << " GB/s" << std::endl;
    std::cout << "v3带宽: " << bandwidth_v3 << " GB/s" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    // 释放资源
    cudaFree(d_in);
    cudaFree(d_out_v1);
    cudaFree(d_out_v2);
    cudaFree(d_out_v3);
    free(h_in);
    free(h_out_v1);
    free(h_out_v2);
    free(h_out_v3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 测试不同大小的矩阵
    test_bandwidth(1024, 1024);
    test_bandwidth(2048, 2048);
    test_bandwidth(4096, 4096);
    test_bandwidth(8192, 8192);
    test_bandwidth(16384, 16384);
    return 0;
}
