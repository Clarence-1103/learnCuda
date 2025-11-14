#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Optimized block-wise reduction function using warp shuffle
__device__ float warpReduceSum(float val) {
    // These shuffle operations are highly efficient and avoid explicit shared memory synchronization
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void meanReductionKernel(const float* input, float* output, int rows, int cols) {
    // Calculate global thread index for processing columns
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (colIdx < cols) {
        // Use shared memory for block-level reduction
        __shared__ float partialSums[256]; // Size should be at least blockDim.y

        // Each thread processes multiple elements along the first dimension (rows)
        float sum = 0.0f;
        for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < rows; i += gridDim.y * blockDim.y) {
            sum += input[i * cols + colIdx];
        }

        // Store the thread's partial sum in shared memory
        partialSums[threadIdx.y] = sum;
        __syncthreads(); // Synchronize all threads in the block

        // Perform block-level reduction within the first warp using warp shuffle
        if (threadIdx.y < 32) { // Only the first warp participates in the final block reduction
            sum = partialSums[threadIdx.y];
            sum = warpReduceSum(sum);
            if (threadIdx.y == 0) {
                // The first thread in the block writes the final block sum to global memory
                // The output array will store the sum for each column, which is the result of reduction along axis 0
                output[blockIdx.x * gridDim.y + blockIdx.y] = sum; // This output setup might need a secondary kernel if gridDim.y is large.
                // For simplicity here, we assume a single large output for each column.
                // A better approach writes to a temporary global buffer and performs a final reduction if needed.
            }
        }
    }
}

// Function to manage kernel launch and data transfers
void computeMeanReduction(const float* h_input, float* h_output, int rows, int cols) {
    float *d_input, *d_output;
    size_t inputSize = rows * cols * sizeof(float);
    size_t outputSize = cols * sizeof(float); // Final output has 'cols' elements

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    
    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    // Kernel configuration: 
    // Grid size in x is cols, as one block is assigned per column for full reduction
    // Block size can be 1D (e.g., 256 threads)
    dim3 threadsPerBlock(256); // Adjust based on architecture for occupancy
    dim3 blocksPerGrid(cols);
    
    // Launch the reduction kernel
    // In a production scenario, for large 'rows', multiple kernel launches might be needed
    // or a persistent kernel approach. This simple example assumes the kernel can finish in one pass.
    meanReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    // Calculate the mean by dividing by the number of rows
    for (int j = 0; j < cols; ++j) {
        h_output[j] /= rows;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    int rows = 1024; // Example dimensions
    int cols = 512;
    float* h_input = new float[rows * cols];
    float* h_output = new float[cols];

    // Initialize input data (example)
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = (float)i;
    }

    computeMeanReduction(h_input, h_output, rows, cols);

    // Print a few results for verification
    printf("Mean of first 5 columns: ");
    for (int j = 0; j < 5; ++j) {
        printf("%f ", h_output[j]);
    }
    printf("\n");

    // Cleanup
    delete[] h_input;
    delete[] h_output;

    return 0;
}

