#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <chrono>
#include <random>

#include <torch/extension.h>
#include <vector>
#include <torch/types.h>

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));
#define CHECK_TORCH_TENSOR_DTYPE(T, expect_type) \
    if (((T).options().dtype() != (expect_type))) { \
        std::cout << "Got input tensor: " << (T).options() << std::endl; \
        std::cout <<"But the kernel should accept tensor with " << (expect_type) << " dtype" << std::endl; \
        throw std::runtime_error("mismatched tensor dtype"); \
    }
#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \

__global__ void hash_retrieval_kernel(
    uint8_t **Q,
    const uint8_t *__restrict__ K,
    int *__restrict__ scores,
    const int *__restrict__ block_table,
    const int *__restrict__ batch_index,
    int dim,
    int block_size,
    int S
) {
    // Q: [batch, dim], the query tensors
    // K: [N, block_size, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    // block_size                一个block多少token
    // gridDim.x                 一共多少个块
    // blockDim.x                每个块多少线程处理dim维度
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    int block_idx = global_x / block_size;
    int t_idx = global_x % block_size;

    if (global_x < S * block_size) {
        int batch_id = batch_index[block_idx];
        int k_index = block_table[block_idx];

        const uint8_t *pQ = Q[batch_id];                               // Q: 查询张量
        const uint8_t *pK = K + (k_index * block_size + t_idx) * dim;  // K: 键张量
        
        int num_tiles = (dim + 16 * blockDim.x - 1) / (16 * blockDim.x);
        int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            int tile_offset = i * (16 * blockDim.x);
            if (tile_offset + local_x * 16 + 16 <= dim) {
                // 加载 16 个 uint8_t 数据（128 位），每次加载 4 个 uint32_t
                const uint4* q4 = reinterpret_cast<const uint4*>(pQ + tile_offset + local_x * 16);
                const uint4* k4 = reinterpret_cast<const uint4*>(pK + tile_offset + local_x * 16);
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

void kvcomp_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index_ranged, torch::Tensor index_sorted, torch::Tensor batch_offset, torch::Tensor workspace){
    // query: a list of ptr
    // repre_cache: a ptr
    int s = q_index.size(0);
    int block_size = repre_cache.size(1);
    int dim = repre_cache.size(2);
    int batch = query_list.size();
    dim3 numThreads = {(unsigned int)(64)};                                       
    dim3 numBlocks = {(unsigned int)(s * block_size)};
    size_t shared_mem = (numThreads.x / 32) * sizeof(int); // warp的数量

    // method 1: use cudaMallocManaged to allocate unified_memory, this perform really good
    uint8_t** Q_ptrs = nullptr;
    cudaMallocManaged(&Q_ptrs, batch * sizeof(uint8_t*));
    for(int i = 0; i < batch; ++i) {
        Q_ptrs[i] = query_list[i].data_ptr<uint8_t>();
    }

    // method 2: copy pointers from host to device, this will spent more time
    // std::vector<uint8_t*> h_Q_ptrs(batch);
    // for(int i = 0; i < batch; ++i) {
    //     h_Q_ptrs[i] = query_list[i].data_ptr<uint8_t>();
    // }
    // uint8_t **Q_ptrs;
    // cuda_check(cudaMalloc(&Q_ptrs, batch * sizeof(uint8_t*)));
    // cuda_check(cudaMemcpy(Q_ptrs, h_Q_ptrs.data(), batch * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    hash_retrieval_kernel<<<numBlocks, numThreads, shared_mem>>>(Q_ptrs, repre_cache.data_ptr<uint8_t>(), score.data_ptr<int>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, block_size, s);
    
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_workspace, temp_bytes,
        score.data_ptr<int>(),  score_sorted.data_ptr<int>(),
        index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
    temp_workspace = workspace.data_ptr<int>();
    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_workspace, temp_bytes,
        score.data_ptr<int>(),  score_sorted.data_ptr<int>(),
        index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
    cuda_check(cudaFree(Q_ptrs));
}

void kvcomp_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);

    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_workspace, temp_bytes,
        score.data_ptr<int>(),  score_out.data_ptr<int>(),
        index.data_ptr<int>(), index_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    // NOTE: don't malloc, just reuse the workspace, but the first call of
    // SortPairs is necesssary to determine the workspace size
    // cuda_check(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();

    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_workspace, temp_bytes,
        score.data_ptr<int>(),  score_out.data_ptr<int>(),
        index.data_ptr<int>(), index_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(kvcomp_retrieval)
    TORCH_BINDING_COMMON_EXTENSION(kvcomp_topk)
}
