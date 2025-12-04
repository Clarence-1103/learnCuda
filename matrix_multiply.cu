#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define M 1024  // A矩阵的行数
#define K 1024  // A矩阵的列数，B矩阵的行数
#define N 1024  // B矩阵的列数

// CUDA kernel：矩阵乘法
__global__ void matrixMulCUDA(float *A, float *B, float *C) {
    // 获取线程在结果矩阵C中的位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断线程是否在矩阵范围内
    if (row < M && col < N) {
        float value = 0;
        // 执行矩阵乘法：C[row][col] = A[row][k] * B[k][col]
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;  // 将计算结果存储到矩阵C
    }
}

// CPU矩阵乘法实现
void matrixMulCPU(float *A, float *B, float *C_ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_ref[i * N + j] = 0;  // 初始化C矩阵的当前元素
            for (int k = 0; k < K; k++) {
                C_ref[i * N + j] += A[i * K + k] * B[k * N + j];  // 计算C_ref[i][j]
            }
        }
    }
}

// 主函数
int main() {
    // 为矩阵A, B, C和C_ref分配内存
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    A = (float*)malloc(size_A);  // 主机内存分配A矩阵
    B = (float*)malloc(size_B);  // 主机内存分配B矩阵
    C = (float*)malloc(size_C);  // 主机内存分配C矩阵（CUDA计算结果）
    C_ref = (float*)malloc(size_C);  // 主机内存分配C矩阵（CPU计算结果）

    // 随机初始化A和B矩阵
    srand(time(0));
    for (int i = 0; i < M * K; i++) {
        A[i] = rand() % 100;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = rand() % 100;
    }

    // 为GPU分配内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 将数据从主机传输到设备
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // 设置CUDA核的块和线程大小
    dim3 threadsPerBlock(16, 16);  // 每个线程块16x16，共256个线程
    dim3 numBlocks((M + 15) / 16, (N + 15) / 16);  // 网格大小，保证覆盖所有矩阵元素

    // --------------------- 性能对比：CUDA部分 ---------------------
    auto start_cuda = std::chrono::high_resolution_clock::now();

    // 执行CUDA矩阵乘法
    matrixMulCUDA<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 从设备复制结果到主机
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    auto stop_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cuda - start_cuda);

    std::cout << "CUDA矩阵乘法执行时间: " << duration_cuda.count() << " 毫秒" << std::endl;

    // --------------------- 性能对比：CPU部分 ---------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // 执行CPU矩阵乘法
    matrixMulCPU(A, B, C_ref);

    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu);

    std::cout << "CPU矩阵乘法执行时间: " << duration_cpu.count() << " 毫秒" << std::endl;

    // --------------------- 结果验证 ---------------------
    bool success = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > 1e-5) {  // 对比CUDA与CPU的计算结果
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "CUDA矩阵乘法与CPU计算结果一致!" << std::endl;
    } else {
        std::cout << "结果不一致!" << std::endl;
    }

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
