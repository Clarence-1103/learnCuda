#include "kvcomp_utils.h"

/**
 * This kernel performs: 
 * score[i] = bitwise_xor(queries[batch_index[i]], repre_cache[block_table[i]]) along the last dimension.
 * The result is then used to compute the Hamming distance by counting the number of 1s in the last dimension.
 * Finally, it finds the minimum Hamming distance along the block_size dimension for each repre_cache.
 * 
 * @param queries: a list of tensors. { [dim] }
 * @param repre_cache: [N, block_size, dim]
 * @param score: [S]
 * @param block_table: [S]
 * @param batch_index: [S]
 */
__global__ void hash_retrieval_kernel(
    uint8_t **queries,
    uint8_t *__restrict__ repre_cache,
    int *__restrict__ scores,
    int *__restrict__ block_table,
    int *__restrict__ batch_index,
    int dim,
    int block_size,
    int S
) {
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    int block_idx = global_x / block_size;
    int t_idx = global_x % block_size;

    if (global_x < S * block_size) {
        int batch_id = batch_index[block_idx];
        int k_index = block_table[block_idx];
        const uint8_t *q = queries[batch_id];
        const uint8_t *k = repre_cache + (k_index * block_size + t_idx) * dim;
        int num_tiles = (dim + blockDim.x - 1) / blockDim.x;
        int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            int tile_offset = i * blockDim.x;
            int idx = tile_offset + local_x;
            if (idx < dim) {
                uint8_t xor_val = q[idx] ^ k[idx];
                sum += __popc(xor_val);
            }
        }

        // Warp-level reduction using __shfl_down_sync
        int local_sum = sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        // Shared memory for warp-level results, assuming blockDim.x is a multiple of 32
        extern __shared__ int shared_sum[];  // Dynamic shared memory, size determined by blockDim.x
        int num_warps = blockDim.x / 32;

        // Store the result of the warp reduction in shared memory
        if (local_x % 32 == 0 && local_x / 32 < num_warps) {
            shared_sum[local_x / 32] = local_sum;
        }
        __syncthreads();

        // Block-level reduction using the results stored in shared memory
        if (local_x < num_warps) {  // Only one thread from each warp will do the next step
            int warp_sum = shared_sum[local_x];

            // Adjust offset dynamically based on blockDim.x
            for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }

            // Only the first thread in the block performs the atomicMin
            if (local_x == 0) {
                atomicMin(&scores[block_idx], warp_sum);
            }
        }
    }
}

__global__ void hash_retrieval_kernel_uint2(
    uint8_t **queries,
    uint8_t *__restrict__ repre_cache,
    int *__restrict__ scores,
    int *__restrict__ block_table,
    int *__restrict__ batch_index,
    int dim,
    int block_size,
    int S
) {
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    int block_idx = global_x / block_size;
    int t_idx = global_x % block_size;

    if (global_x < S * block_size) {
        int batch_id = batch_index[block_idx];
        int k_index = block_table[block_idx];
        const uint8_t *q = queries[batch_id];
        const uint8_t *k = repre_cache + (k_index * block_size + t_idx) * dim;
        int num_tiles = (dim + 8 * blockDim.x - 1) / (8 * blockDim.x);
        int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            int tile_offset = i * (8 * blockDim.x);
            int idx = tile_offset + local_x * 8;
            if (idx + 8 <= dim) {
                // 加载 8 个 uint8_t 数据（64 位），每次加载 2 个 uint32_t
                const uint2* q2 = reinterpret_cast<const uint2*>(q + idx);
                const uint2* k2 = reinterpret_cast<const uint2*>(k + idx);
                uint2 q2_val = *q2;
                uint2 k2_val = *k2;
                uint2 xor_val = make_uint2(q2_val.x ^ k2_val.x, q2_val.y ^ k2_val.y);
                sum += __popc(xor_val.x) + __popc(xor_val.y);
            }
        }

        // Warp-level reduction using __shfl_down_sync
        int local_sum = sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        // Shared memory for warp-level results, assuming blockDim.x is a multiple of 32
        extern __shared__ int shared_sum[];  // Dynamic shared memory, size determined by blockDim.x
        int num_warps = blockDim.x / 32;

        // Store the result of the warp reduction in shared memory
        if (local_x % 32 == 0 && local_x / 32 < num_warps) {
            shared_sum[local_x / 32] = local_sum;
        }
        __syncthreads();

        // Block-level reduction using the results stored in shared memory
        if (local_x < num_warps) {  // Only one thread from each warp will do the next step
            int warp_sum = shared_sum[local_x];

            // Adjust offset dynamically based on blockDim.x
            for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }

            // Only the first thread in the block performs the atomicMin
            if (local_x == 0) {
                atomicMin(&scores[block_idx], warp_sum);
            }
        }
    }
}

__global__ void hash_retrieval_kernel_uint4(
    uint8_t **queries,
    uint8_t *__restrict__ repre_cache,
    int *__restrict__ scores,
    int *__restrict__ block_table,
    int *__restrict__ batch_index,
    int dim,
    int block_size,
    int S
) {
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    int block_idx = global_x / block_size;
    int t_idx = global_x % block_size;

    if (global_x < S * block_size) {
        int batch_id = batch_index[block_idx];
        int k_index = block_table[block_idx];
        const uint8_t *q = queries[batch_id];
        const uint8_t *k = repre_cache + (k_index * block_size + t_idx) * dim;
        int num_tiles = (dim + 16 * blockDim.x - 1) / (16 * blockDim.x);
        int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            int tile_offset = i * (16 * blockDim.x);
            int idx = tile_offset + local_x * 16;
            if (idx + 16 <= dim) {
                // 加载 16 个 uint8_t 数据（128 位），每次加载 4 个 uint32_t
                const uint4* q4 = reinterpret_cast<const uint4*>(q + idx);
                const uint4* k4 = reinterpret_cast<const uint4*>(k + idx);
                uint4 q4_val = *q4;
                uint4 k4_val = *k4;
                uint4 xor_val = make_uint4(q4_val.x ^ k4_val.x, q4_val.y ^ k4_val.y, q4_val.z ^ k4_val.z, q4_val.w ^ k4_val.w);
                sum += __popc(xor_val.x) + __popc(xor_val.y) + __popc(xor_val.z) + __popc(xor_val.w);
            }
        }

        // Warp-level reduction using __shfl_down_sync
        int local_sum = sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        // Shared memory for warp-level results, assuming blockDim.x is a multiple of 32
        extern __shared__ int shared_sum[];  // Dynamic shared memory, size determined by blockDim.x
        int num_warps = blockDim.x / 32;

        // Store the result of the warp reduction in shared memory
        if (local_x % 32 == 0 && local_x / 32 < num_warps) {
            shared_sum[local_x / 32] = local_sum;
        }
        __syncthreads();

        // Block-level reduction using the results stored in shared memory
        if (local_x < num_warps) {  // Only one thread from each warp will do the next step
            int warp_sum = shared_sum[local_x];

            // Adjust offset dynamically based on blockDim.x
            for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }

            // Only the first thread in the block performs the atomicMin
            if (local_x == 0) {
                atomicMin(&scores[block_idx], warp_sum);
            }
        }
    }
}
