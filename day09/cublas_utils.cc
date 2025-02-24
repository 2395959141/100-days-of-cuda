#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


class cublasWrapper {
    private:
        cublasHandle_t   cublas_handle_;
        cublasLtHandle_t cublaslt_handle_;     

        cudaDataType_t Atype_;
        cudaDataType_t Btype_;
        cudaDataType_t Ctype_;
        cudaDataType_t computeType_;   
    
    public:
        cublasWrapper(cublasHandle_t cublas_handle,
                      cublasLtHandle_t cublaslt_handle):
                cublas_handle_(cublas_handle),
                cublaslt_handle_(cublaslt_handle) {}
                      // BaseAllocator* allocator); enable it when we use cublasLt API

        ~cublasWrapper() {}
        void setFP32GemmConfig()
        {
            Atype_       = CUDA_R_32F;
            Btype_       = CUDA_R_32F;
            Ctype_       = CUDA_R_32F;
            computeType_ = CUDA_R_32F;
        }
        void setFP16GemmConfig()
        {
            Atype_       = CUDA_R_16F;
            Btype_       = CUDA_R_16F;
            Ctype_       = CUDA_R_16F;
            computeType_ = CUDA_R_32F;
        }
        //for proj matmul
        void Gemm(cublasOperation_t transa,
                  cublasOperation_t transb,
                  const int         m,
                  const int         n,
                  const int         k,
                  const void*       A,
                  const int         lda,
                  const void*       B,
                  const int         ldb,
                  void*             C,
                  const int         ldc,
                  float            f_alpha = 1.0f,
                  float            f_beta = 0.0f)
        {
            half h_alpha = __float2half(f_alpha);
            half h_beta  = __float2half(f_beta);
            int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0; //之前是CUDA_R_16F
            const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(h_alpha)) : reinterpret_cast<void*>(&f_alpha);
            const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&(h_beta)) : reinterpret_cast<void*>(&f_beta);
            cublasGemmEx(cublas_handle_,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        Atype_,
                        lda,
                        B,
                        Btype_,
                        ldb,
                        beta,
                        C,
                        Ctype_,
                        ldc,
                        computeType_,
                        CUBLAS_GEMM_DEFAULT);
        }

        void stridedBatchedGemm(cublasOperation_t transa,
                                cublasOperation_t transb,
                                const int         m,
                                const int         n,
                                const int         k,
                                const void*       A,
                                const int         lda,
                                const int64_t     strideA,
                                const void*       B,
                                const int         ldb,
                                const int64_t     strideB,
                                void*             C,
                                const int         ldc,
                                const int64_t     strideC,
                                const int         batchCount,
                                float             f_alpha = 1.0f,
                                float             f_beta  = 0.0f)
        {
            int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
            const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(f_alpha)) : reinterpret_cast<const void*>(&f_alpha);
            const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&(f_beta)) : reinterpret_cast<const void*>(&f_beta);
            cublasGemmStridedBatchedEx(cublas_handle_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      Atype_,
                                      lda,
                                      strideA,
                                      B,
                                      Btype_,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      Ctype_,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      computeType_,
                                      CUBLAS_GEMM_DEFAULT);
        }
};
