#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <random>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \


__global__ void extract_repre(const float *key_cache, float *repre_cache, const int *block_table, const int *block_table_2, int block_size, int dim) {
    // key_cache: [N, block_size, dim]
    // repre_cache: [N, 1, dim]
    // block_table: [S]
    // repre_cache[block_table[i]] = mean(key_cache[block_table[i]], 0)
    // NOTE: The last `dimtension` can be processed parallelly. But the
    // `block_size` dim is correlated with each other.
    // So blocks (threads) are tiled for blocks (key_cache)
    // And threads in a block handles different dim

    int idx = blockIdx.x;
    int block_id = block_table[idx];
    int block_id_2 = block_table_2[idx];
    const float* key_ptr = key_cache + block_id * block_size * dim;
    float* repre_ptr = repre_cache + block_id_2 * dim;
    int d = threadIdx.x;
    if (d < dim) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += key_ptr[j * dim + d];
        }
        repre_ptr[d] = sum / block_size;
    }
}



void init_mat(float *mat, int sz){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 5.0f);
    for(int i = 0; i < sz; ++i){
        mat[i] = dist(rng);
    }

}

void host_extract_repre(const float *key_cache, float *repre_cache, const int *block_table, const int *block_table_2, int block_size, int dim, int block_number){
    for(int idx = 0; idx < block_number; ++idx){
        int block_id = block_table[idx];
        int block_id_2 = block_table_2[idx];
        const float* key_ptr = key_cache + block_id * block_size * dim;
        float* repre_ptr = repre_cache + block_id_2 * dim;
        for(int d = 0; d < dim; ++d){
            float sum = 0.0f;
            for(int j = 0; j < block_size; ++j){
                sum += key_ptr[j * dim + d];
            }
            repre_ptr[d] = sum / block_size;
        }
    }
}

int main(){
    int N = 10000;
    int block_size = 128;
    int dim = 1024;
    int block_number = std::min(16 * 65536 / block_size, N);
    int num_layers = 61;
    // host allocations
    float *h_key_cache = (float*)malloc(N * block_size * dim * sizeof(float));
    float *h_repre = (float*)malloc(N * dim * sizeof(float));
    float *h_repre_gpu = (float*)malloc(N * dim * sizeof(float));
    int *h_block_table = (int*)malloc(block_number * sizeof(int));
    int *h_block_table_2 = (int*)malloc(block_number * sizeof(int));

    init_mat(h_key_cache, N * block_size * dim);
    for(int i = 0; i < block_number; ++i){
        h_block_table[i] = (i + 1) * 3 % N;
    }
    for(int i = 0; i < block_number; ++i){
        h_block_table_2[i] = (i + 2) * 5 % N;
    }

    // device allocations
    float *d_key_cache, *d_repre;
    int *d_block_table;
    int *d_block_table_2;
    cuda_check(cudaMalloc(&d_key_cache, N * block_size * dim * sizeof(float)));
    cuda_check(cudaMalloc(&d_repre, N * dim * sizeof(float)));
    cuda_check(cudaMalloc(&d_block_table, block_number * sizeof(int)));
    cuda_check(cudaMalloc(&d_block_table_2, block_number * sizeof(int)));
    cuda_check(cudaMemcpy(d_key_cache, h_key_cache, N * block_size * dim * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_block_table, h_block_table, block_number * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_block_table_2, h_block_table_2, block_number * sizeof(int), cudaMemcpyHostToDevice));

    // warm‐up
    int threads = dim;
    int blocks = block_number;
    for(int i = 0; i < 10; ++i){
        extract_repre<<<blocks, threads>>>(d_key_cache, d_repre, d_block_table, d_block_table_2, block_size, dim);
    }

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i = 0; i < num_layers; ++i){
        extract_repre<<<blocks, threads>>>(d_key_cache, d_repre, d_block_table, d_block_table_2, block_size, dim);
    }
    cudaEventRecord(stop, 0);
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaEventSynchronize(stop));
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("extract %d blocks (dim = %d) spent on extract_repre: %f ms\n", block_number, dim, ms);

    // copy back and verify
    cuda_check(cudaMemcpy(h_repre_gpu, d_repre, N * dim * sizeof(float), cudaMemcpyDeviceToHost));
    host_extract_repre(h_key_cache, h_repre, h_block_table, h_block_table_2, block_size, dim, block_number);
    float avg_err = 0.0f;
    for(int i = 0; i < block_number; ++i){
        auto cur_err = 0.0f;
        auto block_idx = h_block_table_2[i];
        for(int j = 0; j < dim; ++j){
            cur_err += (h_repre[block_idx * dim + j] - h_repre_gpu[block_idx * dim + j]);
            // if((i % 10) == 0 && j == dim / 2){
            //     printf("host %f vs device %f\n", h_repre[block_idx * dim + j], h_repre_gpu[block_idx * dim + j]);
            // }
        }
        // printf("block %d err %f\n", h_block_table_2[i], cur_err);
        avg_err += cur_err;
    }
    avg_err /= (block_number * dim);
    printf("avg error: %f\n", avg_err);
    return 0;
}

