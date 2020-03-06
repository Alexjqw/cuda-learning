#include"../include/wta_kernel.hpp"
#include<iostream>

namespace srs{

    static constexpr unsigned int WARPS_PER_BLOCK = 8u;
    //static constexpr unsigned int WARP_SIZE = 32u;
    static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;


    __device__ inline uint32_t pack_cost_index(uint32_t cost, uint32_t index) {
        union {
            uint32_t uint32;
            ushort2 uint16x2;
        } u;
        u.uint16x2.x = static_cast<uint16_t>(index);
        u.uint16x2.y = static_cast<uint16_t>(cost);
        return u.uint32;
    }

    __device__ uint32_t unpack_cost(uint32_t packed) {
        return packed >> 16;
    }

    __device__ int unpack_index(uint32_t packed) {
        return packed & 0xffffu;
    }

    using ComputeDisparity = uint32_t(*)(uint32_t, uint32_t, uint8_t*);

    __device__ inline uint32_t compute_disparity_normal(uint32_t disp, uint32_t cost = 0, uint8_t* smem = nullptr)
    {
        return disp;
    }

    template <size_t MAX_DISPARITY>
    __device__ inline uint32_t compute_disparity_subpixel(uint32_t disp, uint32_t cost, uint8_t* smem)
    {
        int subp = disp;
        subp <<= 4;
        if (disp > 0 && disp < MAX_DISPARITY - 1) {
            const int left = smem[disp - 1];
            const int right = smem[disp + 1];
            const int numer = left - right;
            const int denom = left - 2 * cost + right;
            subp += ((numer << 4) + denom) / (2 * denom);
        }
        return subp;
    }



    template <unsigned int MAX_DISPARITY, ComputeDisparity compute_disparity = compute_disparity_normal>
    __global__ void winner_takes_all_kernel(
        output_type *left_dest,
        const census_type *src,
        int width,
        int height,
        int pitch,
        float uniqueness)
    {
        static const unsigned int ACCUMULATION_PER_THREAD = 16u;
        static const unsigned int REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE;
        static const unsigned int ACCUMULATION_INTERVAL = ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD;
        static const unsigned int UNROLL_DEPTH =
            (REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL)
            ? REDUCTION_PER_THREAD
            : ACCUMULATION_INTERVAL;


        const unsigned int warp_id = threadIdx.x / WARP_SIZE;
        const unsigned int lane_id = threadIdx.x % WARP_SIZE;

        const unsigned int y = blockIdx.x * WARPS_PER_BLOCK + warp_id;
        src += y * MAX_DISPARITY * width;
        left_dest += y * pitch;


        if (y >= height) {
            return;
        }
        __shared__ uint8_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];
        for (unsigned int x0 = 0; x0 < width; x0 += UNROLL_DEPTH) {
#pragma unroll
            for (unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1) {
                if (x1 % ACCUMULATION_INTERVAL == 0) {
                    const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                    const unsigned int k_hi = k / MAX_DISPARITY;
                    const unsigned int k_lo = k % MAX_DISPARITY;
                    const unsigned int x = x0 + x1 + k_hi;
                    if (x < width) {
                        const unsigned int offset = x * MAX_DISPARITY + k_lo;
                      /*  copy_data<ACCUMULATION_PER_THREAD>(
                            &smem_cost_sum[warp_id][k_hi][k_lo], &src[offset]);*/
                        uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                        load_uint8_vector<ACCUMULATION_PER_THREAD>(
                            load_buffer, &src[offset]);
                        store_uint8_vector<ACCUMULATION_PER_THREAD>(
                            &smem_cost_sum[warp_id][k_hi][k_lo], load_buffer);
                    }
#if CUDA_VERSION >= 9000
                    __syncwarp();
#else
                    __threadfence_block();
#endif
                }
                const unsigned int x = x0 + x1;
                if (x < width) {
                    // Load sum of costs
                    const unsigned int smem_x = x1 % ACCUMULATION_INTERVAL;
                    const unsigned int k0 = lane_id * REDUCTION_PER_THREAD;
                    uint32_t local_cost_sum[REDUCTION_PER_THREAD];
                    load_uint8_vector<REDUCTION_PER_THREAD>(
                        local_cost_sum, &smem_cost_sum[warp_id][smem_x][k0]);
                    // Pack sum of costs and dispairty
                    uint32_t local_packed_cost[REDUCTION_PER_THREAD];
                    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) {
                        local_packed_cost[i] = pack_cost_index(local_cost_sum[i], k0 + i);
                    }
                    // Update left
                    uint32_t best = 0xffffffffu;
                    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) {
                        best = min(best, local_packed_cost[i]);
                    }
                    best = subgroup_min<WARP_SIZE>(best, 0xffffffffu);

#pragma unroll
                    // Resume updating left to avoid execution dependency
                    const uint32_t bestCost = unpack_cost(best);
                    const int bestDisp = unpack_index(best);
                    bool uniq = true;
                    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) {
                        const uint32_t x = local_packed_cost[i];
                        const bool uniq1 = unpack_cost(x) * uniqueness >= bestCost;
                        const bool uniq2 = abs(unpack_index(x) - bestDisp) <= 1;
                        uniq &= uniq1 || uniq2;
                    }
                    uniq = subgroup_and<WARP_SIZE>(uniq, 0xffffffffu);
                    if (lane_id == 0) {
                        left_dest[x] = uniq ? compute_disparity(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]) : INVALID_DISP;
                    }
                }
            }
        }

    }


    void enqueue_winner_takes_all(
        output_type *left_dest,
        const census_type *src,
        int width,
        int height,
        int pitch,        
        float uniqueness,
        bool subpixel,
        cudaStream_t stream)
    {
        const int gdim =
            (height + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int bdim = BLOCK_SIZE;
        if (subpixel) {
            winner_takes_all_kernel<64, compute_disparity_subpixel<64>> << <gdim, bdim, 0, stream >> > (
                left_dest, src, width, height, pitch, uniqueness);
        }
        else
        {
            winner_takes_all_kernel<64, compute_disparity_normal> << <gdim, bdim, 0, stream >> > (
                left_dest, src, width, height, pitch, uniqueness);

        }
    }

}

