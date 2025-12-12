#include <stdio.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>
#include <torch/types.h>

namespace py = pybind11;

#define STRINGFY(func) #func

#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func))

#define CHECK_TORCH_TENSOR_DTYPE(T, expect_type) \
    if (((T).options().dtype() != (expect_type))) { \
        std::cout << "Got input tensor: " << (T).options() << std::endl; \
        std::cout <<"But the kernel should accept tensor with " << (expect_type) << " dtype" << std::endl; \
        throw std::runtime_error("mismatched tensor dtype"); \
    }

#define CUDA_CHECK(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \
