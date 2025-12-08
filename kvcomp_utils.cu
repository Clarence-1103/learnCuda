#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <chrono>
#include <random>

#include <torch/extension.h>
#include <vector>
#include <torch/types.h>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \


__global__ void hash_retrieval_kernel(const uint8_t *__restrict__ Q, const uint8_t *__restrict__ K, int * scores, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int B, int block_size, int S) {
    // Q: [batch, dim], the query tensors
    // K: [N, block_size, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S) {
        int batch_id = batch_index[idx];
        int k_index = block_table[idx];
        int score = UINT16_MAX;
        for(size_t t_idx = 0; t_idx < block_size; ++t_idx){
            const uint8_t *pQ = Q + batch_id * dim;
            const uint8_t *pK = K + (k_index * block_size + t_idx) * dim;
            int sum = 0;
            #pragma unroll 8
            for(int j = 0; j < dim; ++j){
                int xor_val = pQ[j] ^ pK[j];
                sum += __popc(xor_val);
            }

            if(sum < score){
                score = sum;
                if (score == 0) {
                    break;
                }
            }
        }
        scores[idx] = score;
    }
}

__global__ void hash_retrieval_kernel_1(const uint8_t *__restrict__ Q, const uint8_t *__restrict__ K, int * scores, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int B, int block_size, int S) {
    // Q: [batch, dim], the query tensors
    // K: [N, block_size, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S) {
        int batch_id = batch_index[idx];
        int k_index = block_table[idx];
        int score = UINT16_MAX;

        for (size_t t_idx = 0; t_idx < block_size; ++t_idx) {
            const uint8_t *pQ = Q + batch_id * dim;  // Q: 查询张量
            const uint8_t *pK = K + (k_index * block_size + t_idx) * dim;  // K: 键张量
            int sum = 0;

            // 使用 128 位数据块（16个 uint8_t）
            #pragma unroll 8
            for (int j = 0; j < dim; j += 16) {  // 每次处理 16 个 uint8_t
                // 加载 16 个 uint8_t 数据（128 位），每次加载 4 个 uint32_t
                const uint4* q4 = reinterpret_cast<const uint4*>(pQ + j);  // 加载 4 个 uint32_t
                const uint4* k4 = reinterpret_cast<const uint4*>(pK + j);  // 加载 4 个 uint32_t

                // 进行 XOR 操作
                uint4 xor_val = make_uint4(q4->x ^ k4->x, q4->y ^ k4->y, q4->z ^ k4->z, q4->w ^ k4->w);
                // 计算 XOR 结果中的 1 的个数（汉明距离）
                sum += __popc(xor_val.x) + __popc(xor_val.y) + __popc(xor_val.z) + __popc(xor_val.w);
            }

            // 计算最小的汉明距离
            if (sum < score) {
                score = sum;
                if (score == 0) {
                    break;
                }
            }
        }
        scores[idx] = score;
    }
}

__global__ void hash_retrieval_kernel_2(const uint8_t *__restrict__ Q, const uint8_t *__restrict__ K, int * scores, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int B, int block_size, int S) {
    // Q: [batch, dim], the query tensors
    // K: [N, block_size, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S) {
        int batch_id = batch_index[idx];
        int k_index = block_table[idx];
        const uint8_t *pQ = Q + batch_id * dim;  // Q: 查询张量
        int score = UINT16_MAX;
        for (size_t t_idx = 0; t_idx < block_size; ++t_idx) {
            
            const uint8_t *pK = K + (k_index * block_size + t_idx) * dim;  // K: 键张量
            int sum = 0;

            // 使用 128 位数据块（16个 uint8_t）
            #pragma unroll 8
            for (int j = 0; j < dim; j += 16) {  // 每次处理 16 个 uint8_t
                // 加载 16 个 uint8_t 数据（128 位），每次加载 4 个 uint32_t
                const uint4* q4 = reinterpret_cast<const uint4*>(pQ + j);  // 加载 4 个 uint32_t
                const uint4* k4 = reinterpret_cast<const uint4*>(pK + j);  // 加载 4 个 uint32_t

                uint4 q4_val = *q4;  // 将加载的数据保存在寄存器中
                uint4 k4_val = *k4;  // 将加载的数据保存在寄存器中
                uint4 xor_val = make_uint4(q4_val.x ^ k4_val.x, q4_val.y ^ k4_val.y, q4_val.z ^ k4_val.z, q4_val.w ^ k4_val.w);
                // 计算 XOR 结果中的 1 的个数（汉明距离）
                sum += __popc(xor_val.x) + __popc(xor_val.y) + __popc(xor_val.z) + __popc(xor_val.w);
            }

            // 计算最小的汉明距离
            if (sum < score) {
                score = sum;
                if (score == 0) {
                    break;
                }
            }
        }
        scores[idx] = score;
    }
}

__global__ void hash_retrieval_kernel_3(const uint8_t *__restrict__ Q, const uint8_t *__restrict__ K, int * scores, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int B, int block_size, int S) {
    // Q: [batch, dim], the query tensors
    // K: [N, block_size, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    // gridDim.x = S * block_size 一共多少个块
    // blockDim.x = 32           每个块多少线程处理dim维度
    extern __shared__ int local_score[]; // 动态分配共享内存
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    int block_idx = global_x / block_size;
    int t_idx = global_x % block_size;

    if (global_x < S * block_size) {
        int batch_id = batch_index[block_idx];
        int k_index = block_table[block_idx];

        const uint8_t *pQ = Q + batch_id * dim;  // Q: 查询张量
        const uint8_t *pK = K + (k_index * block_size + t_idx) * dim;  // K: 键张量
        
        // blockDim.x = 8；uint4就是 16 * blockDim.x = 128
        int num_tiles = (dim + 16 * blockDim.x - 1) / (16 * blockDim.x); // 每个tile处理128个uint_8
        int sum = 0;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (16 * blockDim.x);   // 一次偏移128个uint8，
            // 总共有32个线程，每128个uint_8偏移一次，总共1024位， loacl_max=31，32个线程
            // 每个线程处理16个uint_8
            if(tile_offset + local_x * 16 + 16 <= dim){
                // 加载 16 个 uint8_t 数据（128 位），每次加载 4 个 uint32_t
                const uint4* q4 = reinterpret_cast<const uint4*>(pQ + tile_offset + local_x * 16);
                const uint4* k4 = reinterpret_cast<const uint4*>(pK + tile_offset + local_x * 16);
                uint4 q4_val = *q4;
                uint4 k4_val = *k4;
                uint4 xor_val = make_uint4(q4_val.x ^ k4_val.x, q4_val.y ^ k4_val.y, q4_val.z ^ k4_val.z, q4_val.w ^ k4_val.w);
                sum += __popc(xor_val.x) + __popc(xor_val.y) + __popc(xor_val.z) + __popc(xor_val.w);
            }
        }

        local_score[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score[local_x] = local_score[local_x] + local_score[local_x + i];
            }
            __syncthreads();
        }

        if (local_x == 0) {
            atomicMin(&scores[block_idx], local_score[local_x]);  // Only the first thread performs the atomicMin
        }
    }
}

__global__ void hash_retrieval_kernel_4(
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
        
        int num_tiles = (dim + 16 * blockDim.x - 1) / (16 * blockDim.x); // 每个tile处理128个uint_8
        int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            int tile_offset = i * (16 * blockDim.x);   // 一次偏移128个uint8，
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

        // Warp-level reduction using shuffle
        int local_sum = sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        // 使用共享内存存储每个 warp 的计算结果
        extern __shared__ int warp_sums[]; // 动态分配共享内存

        int warp_id = local_x / 32;  // 每个 warp 有 32 个线程

        // 每个 warp 线程执行完后，将 sum 存入共享内存
        if (local_x % 32 == 0) {
            warp_sums[warp_id] = local_sum;
        }
        __syncthreads();  // 等待所有 warp 完成

        // Warp 之间的归约：块内的线程将 warp 的结果进行归约
        if (warp_id == 0) {
            int warp_total = 0;
            // 对当前块的所有 warp 的 sum 进行汇总
            for (int i = 0; i < 32; i++) {
                warp_total += warp_sums[i];
            }

            // 将每个块的结果存储到全局结果数组中
            if (local_x == 0) {
                atomicMin(&scores[block_idx], warp_total);
            }
        }
    }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>   // SSE/AVX
    #include <nmmintrin.h>   // POPCNT (SSE4.2)
#endif

#define VEC_SIZE 16
int vec_per_dim_;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

using vec16u = uint8x16_t;

static inline vec16u vec_loadu16(const uint8_t* p) {
    return vld1q_u8(p);
}

static inline vec16u vec_xor(vec16u a, vec16u b) {
    return veorq_u8(a, b);
}

static inline uint16_t vec_sum_u8(vec16u v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_u8(v);
#else
    uint16x8_t s16 = vpaddlq_u8(v);
    uint32x4_t s32 = vpaddlq_u16(s16);
    uint64x2_t s64 = vpaddlq_u32(s32);
    return (uint16_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
}

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    vec16u va = vec_loadu16(a);
    vec16u vb = vec_loadu16(b);
    vec16u vx = vec_xor(va, vb);
    vec16u pc = vcntq_u8(vx);
    return vec_sum_u8(pc);
}

static inline uint16_t vec_popcnt_xor_sum16_vec(vec16u qa, const uint8_t* b) {
    vec16u vb = vec_loadu16(b);
    vec16u vx = vec_xor(qa, vb);
    vec16u pc = vcntq_u8(vx);
    return vec_sum_u8(pc);
}

void print_uint8x16(uint8x16_t vec) {
    uint8_t array[16];
    vst1q_u8(array, vec);
    for (int i = 0; i < 16; ++i) {
        std::cout << static_cast<int>(array[i]) << " ";
    }
    std::cout << std::endl;
}

#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

using vec16u = __m128i;

static inline vec16u vec_loadu16(const uint8_t* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

static inline vec16u vec_xor(vec16u a, vec16u b) {
    return _mm_xor_si128(a, b);
}

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i vx = _mm_xor_si128(va, vb);

    uint64_t lo, hi;
#if defined(__SSE4_1__)
    lo = static_cast<uint64_t>(_mm_extract_epi64(vx, 0));
    hi = static_cast<uint64_t>(_mm_extract_epi64(vx, 1));
#else
    alignas(16) uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vx);
    lo = tmp[0];
    hi = tmp[1];
#endif
    return (uint16_t)(__builtin_popcountll(lo) + __builtin_popcountll(hi));
}

static inline uint16_t vec_popcnt_xor_sum16_vec(vec16u qa, const uint8_t* b) {
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i vx = _mm_xor_si128(qa, vb);

    uint64_t lo, hi;
#if defined(__SSE4_1__)
    lo = static_cast<uint64_t>(_mm_extract_epi64(vx, 0));
    hi = static_cast<uint64_t>(_mm_extract_epi64(vx, 1));
#else
    alignas(16) uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vx);
    lo = tmp[0];
    hi = tmp[1];
#endif
    return (uint16_t)(__builtin_popcountll(lo) + __builtin_popcountll(hi));
}

#else   

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    uint16_t s = 0;
    for (int i = 0; i < 16; ++i)
        s += __builtin_popcount((unsigned)(a[i] ^ b[i]));
    return s;
}

#endif

void hash_retrieval_host(uint8_t *Q, uint8_t *K, int *scores, int *block_table, int *batch_index, int dim, int B, int block_size, int S){
    for(int i = 0; i < S; ++i){
        int batch_id = batch_index[i];
        int k_index = block_table[i];
        int score = UINT16_MAX;

        const uint8_t* q_ptr = Q + batch_id * dim;
        const uint8_t* base_idx_ptr = K + k_index * block_size * dim;

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        // 1.预加载 query 向量
        vec16u q_vecs[vec_per_dim_]; // 存储query向量
        for (size_t v = 0; v < vec_per_dim_; ++v) {
            q_vecs[v] = vec_loadu16(q_ptr + v * VEC_SIZE);
        }
#endif
        // 3.内层向量化计算
        for (size_t t_idx = 0; t_idx < block_size; ++t_idx) {
            int sum = 0;
            const uint8_t* k_base = base_idx_ptr + t_idx * dim;

            // 计算每个向量的相似度
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            for (size_t v = 0; v < vec_per_dim_; ++v) {
                sum += vec_popcnt_xor_sum16_vec(
                    q_vecs[v],
                    k_base + v * VEC_SIZE
                );
            }
#else
            for (size_t v = 0; v < vec_per_dim_; ++v) {
                sum += vec_popcnt_xor_sum16(
                    q_ptr  + v * VEC_SIZE,
                    k_base + v * VEC_SIZE
                );
            }
#endif

            if (sum < score) {
                score = sum;
                if (score == 0) {
                    break;
                }
            }
        }
        scores[i] = score;
    }
}

void init_mat(uint8_t *mat, int sz) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 255);  // 生成 0 到 255 之间的随机数
    for (int i = 0; i < sz; ++i) {
        mat[i] = dist(rng);  // 填充 mat 数组
    }
}

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));
#define CHECK_TORCH_TENSOR_DTYPE(T, expect_type) \
if (((T).options().dtype() != (expect_type))) { \
    std::cout << "Got input tensor: " << (T).options() << std::endl; \
    std::cout <<"But the kernel should accept tensor with " << (expect_type) << " dtype" << std::endl; \
    throw std::runtime_error("mismatched tensor dtype"); \
}

void kvcomp_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score){
    // query: a list of ptr
    // repre_cache: a ptr
    CHECK_TORCH_TENSOR_DTYPE(query_list[0], torch::kUInt8);
    CHECK_TORCH_TENSOR_DTYPE(repre_cache, torch::kUInt8);
    int s = q_index.size(0);
    int block_size = repre_cache.size(1);
    int dim = repre_cache.size(2);
    int batch = query_list.size();
    dim3 numThreads = {(unsigned int)(32)};
    dim3 numBlocks = {(unsigned int) s};
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
    
    hash_retrieval_kernel_4<<<numBlocks, numThreads, shared_mem>>>(Q_ptrs, repre_cache.data_ptr<uint8_t>(), score.data_ptr<int>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, block_size, s);
    cuda_check(cudaFree(Q_ptrs));
}

void kvcomp_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, size_t K){
    CHECK_TORCH_TENSOR_DTYPE(offsets, torch::kInt32);
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp, temp_bytes,
        score.data_ptr<int>(),  score_out.data_ptr<int>(),
        index.data_ptr<int>(), index_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    cuda_check(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp, temp_bytes,
        score.data_ptr<int>(),  score_out.data_ptr<int>(),
        index.data_ptr<int>(), index_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(kvcomp_retrieval)
    TORCH_BINDING_COMMON_EXTENSION(kvcomp_topk)
}

// int main(){
//     uint8_t *h_Q, *h_K;
//     int B ;
//     int seq_len;
//     int dim;
//     scanf("%d%d%d", &B, &seq_len, &dim);
//     vec_per_dim_ = dim / VEC_SIZE;  // data_每个值类型uint8_t,组成8*16_t进行simd加速

//     int N = 4000; // 总共有多少个blocks
//     int block_size = 128;
//     h_Q = (uint8_t*)malloc(B * dim * sizeof(uint8_t));
//     h_K = (uint8_t*)malloc(N * block_size * dim * sizeof(uint8_t));

//     init_mat(h_Q, B * dim);
//     init_mat(h_K, N * block_size * dim);

//     int total_kv_len = 0; //total_kv_len = S = B * num_blocks 所有batch用的number blocks之和
//     int *h_kv_len;
//     h_kv_len = (int*)malloc(B * sizeof(int));
//     int *kv_start_offsets;
//     kv_start_offsets = (int*)malloc((B+1) * sizeof(int));
//     int kv_len_each = (seq_len / block_size);   // 实际每个batch用的number blocks
//     for(int i = 0; i < B; ++i){
//         h_kv_len[i] = kv_len_each;
//         kv_start_offsets[i] = total_kv_len;
//         total_kv_len += h_kv_len[i];
//     }
//     kv_start_offsets[B] = total_kv_len;
//     int *h_score;
//     h_score = (int*)malloc(total_kv_len * sizeof(int));   //total_kv_len = S = B * num_blocks 所有batch用的number blocks之和

//     int *block_table;
//     block_table = (int*)malloc(total_kv_len * sizeof(int));
//     for(int i = 0; i < total_kv_len; ++i){
//         block_table[i] = i * 5 % N;     // just a random mapping from S to N
//     }

//     // 这段代码的目的是根据批次的开始和结束偏移量（kv_start_offsets），将每个元素索引 i 分配到对应的批次 j
//     int *batch_index;
//     batch_index = (int*)malloc(total_kv_len * sizeof(int));
//     for(int i = 0, j = 0; i < total_kv_len; ++i){
//         if(i < kv_start_offsets[j+1] && i >= kv_start_offsets[j]){
//             batch_index[i] = j;
//         }
//         else{
//             ++j;
//             batch_index[i] = j;
//         }
//     }

//     uint8_t *d_Q, *d_K;
//     cuda_check(cudaMalloc(&d_Q, sizeof(uint8_t) * B * dim));
//     cuda_check(cudaMalloc(&d_K, sizeof(uint8_t) * N * block_size * dim));

//     int *d_score;
//     cuda_check(cudaMalloc(&d_score, sizeof(int) * total_kv_len)); // for kernel_3
//     // 将内存填充为 0x7F（INT_MAX 的最高字节值）
//     cuda_check(cudaMemset(d_score, 0x7F, sizeof(int) * total_kv_len));
//     cuda_check(cudaMemcpy(d_Q, h_Q, sizeof(uint8_t) * B * dim, cudaMemcpyHostToDevice));
//     cuda_check(cudaMemcpy(d_K, h_K, sizeof(uint8_t) * N * block_size * dim, cudaMemcpyHostToDevice));

//     int *d_block_table, *d_batch_index;
//     cuda_check(cudaMalloc(&d_block_table, sizeof(int) * total_kv_len));
//     cuda_check(cudaMemcpy(d_block_table, block_table, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));
//     cuda_check(cudaMalloc(&d_batch_index, sizeof(int) * total_kv_len));
//     cuda_check(cudaMemcpy(d_batch_index, batch_index, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));

//     dim3 numThreads = {(unsigned int)(128)};                            // 128 threads per block = block_size
//     dim3 numBlocks = {(unsigned int)(total_kv_len)};                    // total_kv_len =  所有batch用的number blocks之和
    
//     // warm-up
//     for (int i = 0; i < 10; ++i){
//         hash_retrieval_kernel<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//         hash_retrieval_kernel_1<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//         hash_retrieval_kernel_2<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//         size_t bytes =  numThreads.x * sizeof(int);
//         hash_retrieval_kernel_3<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//         size_t shared_mem = (numThreads.x / 32) * sizeof(int); // warp的数量
//         hash_retrieval_kernel_4<<<numBlocks, numThreads, shared_mem>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, block_size, total_kv_len);
//     }

//     cudaEvent_t start, stop, start_1, stop_1, start_2, stop_2, start_3, stop_3, start_4, stop_4;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventCreate(&start_1);
//     cudaEventCreate(&stop_1);
//     cudaEventCreate(&start_2);
//     cudaEventCreate(&stop_2);
//     cudaEventCreate(&start_3);
//     cudaEventCreate(&stop_3);
//     cudaEventCreate(&start_4);
//     cudaEventCreate(&stop_4);
    
//     // v0
//     cudaEventRecord(start, 0);
//     hash_retrieval_kernel<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//     cudaEventRecord(stop, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop));
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time spent on hash_retrieval_kernel: %f ms\n", milliseconds);
    
//     // v1
//     cudaEventRecord(start_1, 0);
//     hash_retrieval_kernel_1<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//     cudaEventRecord(stop_1, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop_1));
//     float milliseconds_1 = 0;
//     cudaEventElapsedTime(&milliseconds_1, start_1, stop_1);
//     printf("Time spent on hash_retrieval_kernel_1: %f ms\n", milliseconds_1);

//     // v2
//     cudaEventRecord(start_2, 0);
//     hash_retrieval_kernel_2<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//     cudaEventRecord(stop_2, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop_2));
//     float milliseconds_2 = 0;
//     cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);
//     printf("Time spent on hash_retrieval_kernel_2: %f ms\n", milliseconds_2);

//     // v3
//     cudaEventRecord(start_3, 0);
//     size_t bytes =  numThreads.x * sizeof(int);
//     hash_retrieval_kernel_3<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, block_size, total_kv_len);
//     cudaEventRecord(stop_3, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop_3));
//     float milliseconds_3 = 0;
//     cudaEventElapsedTime(&milliseconds_3, start_3, stop_3);
//     printf("Time spent on hash_retrieval_kernel_3: %f ms\n", milliseconds_3);

//     // v4
//     cudaEventRecord(start_4, 0);
//     size_t shared_mem = (numThreads.x / 32) * sizeof(int); // warp的数量
//     hash_retrieval_kernel_4<<<numBlocks, numThreads, shared_mem>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, block_size, total_kv_len);
//     cudaEventRecord(stop_4, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop_4));
//     float milliseconds_4 = 0;
//     cudaEventElapsedTime(&milliseconds_4, start_4, stop_4);
//     printf("Time spent on hash_retrieval_kernel_4: %f ms\n", milliseconds_4);


//     int *h_score_gpu;
//     h_score_gpu = (int*)malloc(total_kv_len * sizeof(int));
//     cuda_check(cudaMemcpy(h_score_gpu, d_score, total_kv_len * sizeof(int), cudaMemcpyDeviceToHost));
    
//     for (int i = 0; i < 10; ++i){
//         hash_retrieval_host(h_Q, h_K, h_score, block_table, batch_index, dim, B, block_size, total_kv_len);
//     }
    
//     auto h_start = std::chrono::high_resolution_clock::now();
//     hash_retrieval_host(h_Q, h_K, h_score, block_table, batch_index, dim, B, block_size, total_kv_len);
//     auto h_stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(h_stop - h_start);
//     double duration_ms = (double)duration.count() / 1000000.0; // 纳秒 -> 毫秒
//     printf("Time spent on hash retrieval_host: %.6f ms\n", duration_ms);  // 打印毫秒，保留 6 位小数

//     float eps = 1e-3;
//     float avg_error = 0.0f;
//     for(int i = 0; i < total_kv_len; ++i){
//         float diff = fabs(h_score[i] - h_score_gpu[i]);
//         avg_error += diff;
//         if(diff > eps){
//             printf("not ok @%d!!! %d vs %d, err %f\n", i, h_score[i], h_score_gpu[i], diff);
//         }
//     }
//     avg_error = avg_error / total_kv_len;
//     printf("avg error: %f\n", avg_error);

//     return 0;
// }
