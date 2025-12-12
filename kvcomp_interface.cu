#include "kvcomp_utils.h"
#include "kvcomp_kernels.cu"

struct RetrievalInputTensor{
    py::list query_list;
    torch::Tensor repre_cache;
    torch::Tensor q_index;
    torch::Tensor repre_index;
    torch::Tensor batch_offset;
    torch::Tensor workspace;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    torch::Tensor score_sorted;
    torch::Tensor index_ranged;
    torch::Tensor index_sorted;
};

void kvcomp_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
    auto query_list = input.query_list;
    auto repre_cache = input.repre_cache;
    auto q_index = input.q_index;
    auto repre_index = input.repre_index;
    auto batch_offset = input.batch_offset;
    auto workspace = input.workspace;

    auto score = output.score;
    auto score_sorted = output.score_sorted;
    auto index_ranged = output.index_ranged;
    auto index_sorted = output.index_sorted;

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
        auto q_tensor = query_list[i].cast<torch::Tensor>();
        Q_ptrs[i] = q_tensor.data_ptr<uint8_t>();
    }

    // method 2: copy pointers from host to device, this will spent more time
    // std::vector<uint8_t*> h_Q_ptrs(batch);
    // for(int i = 0; i < batch; ++i) {
    //     h_Q_ptrs[i] = query_list[i].data_ptr<uint8_t>();
    // }
    // uint8_t **Q_ptrs;
    // cuda_check(cudaMalloc(&Q_ptrs, batch * sizeof(uint8_t*)));
    // cuda_check(cudaMemcpy(Q_ptrs, h_Q_ptrs.data(), batch * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    
    if (dim % 16 != 0 && dim % 8 == 0) {
        hash_retrieval_kernel_uint2<<<numBlocks, numThreads, shared_mem>>>(Q_ptrs, repre_cache.data_ptr<uint8_t>(), score.data_ptr<int>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, block_size, s);
    } else if (dim % 16 == 0) {
        hash_retrieval_kernel_uint4<<<numBlocks, numThreads, shared_mem>>>(Q_ptrs, repre_cache.data_ptr<uint8_t>(), score.data_ptr<int>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, block_size, s);
    } else {
        hash_retrieval_kernel<<<numBlocks, numThreads, shared_mem>>>(Q_ptrs, repre_cache.data_ptr<uint8_t>(), score.data_ptr<int>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, block_size, s);
    }
    
    CUDA_CHECK(cudaFree(Q_ptrs));
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
    m.doc() = "KVCOMP cuda kernels for topk block retrieval";
    py::class_<RetrievalInputTensor>(m, "RetrievalInputTensor")
        .def(py::init<>())
        .def_readwrite("query_list", &RetrievalInputTensor::query_list)
        .def_readwrite("repre_cache", &RetrievalInputTensor::repre_cache)
        .def_readwrite("q_index", &RetrievalInputTensor::q_index)
        .def_readwrite("repre_index", &RetrievalInputTensor::repre_index)
        .def_readwrite("batch_offset", &RetrievalInputTensor::batch_offset)
        .def_readwrite("workspace", &RetrievalInputTensor::workspace);

    py::class_<RetrievalOutputTensor>(m, "RetrievalOutputTensor")
        .def(py::init<>())
        .def_readwrite("score", &RetrievalOutputTensor::score)
        .def_readwrite("score_sorted", &RetrievalOutputTensor::score_sorted)
        .def_readwrite("index_ranged", &RetrievalOutputTensor::index_ranged)
        .def_readwrite("index_sorted", &RetrievalOutputTensor::index_sorted);

    TORCH_BINDING_COMMON_EXTENSION(kvcomp_retrieval);
    TORCH_BINDING_COMMON_EXTENSION(kvcomp_topk);
}
