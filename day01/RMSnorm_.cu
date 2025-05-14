
#define WARP_SIZE 32



template<const int KwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
    for (int i = Kwarpsize >> 1; i >= 1; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_fp16(float val) {
    constexpr int NUM_WRAPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float shared[NUM_WRAPS];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    static __shared__ float shared[NUM_WRAPS];

    val = warp_reduce_sum_fp32(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    
}