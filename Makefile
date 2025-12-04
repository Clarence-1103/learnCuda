clean:
	rm -rf ./matmul ./mat_transpose ./softmax ./copy_compute_test
matmul:
	nvcc -o ./matmul ./matmul.cu
mat_transpose:
	nvcc -o ./mat_transpose ./mat_transpose.cu
softmax:
	nvcc -o ./softmax ./softmax.cu
copy_compute_test:
	nvcc -o ./copy_compute_test ./copy_compute_test.cu
retrieval_kernel:
	nvcc -o ./retrieval_kernel ./retrieval_kernel.cu
topk_kernel:
	nvcc -o ./topk_kernel ./topk_kernel.cu
matrix_multiply:
	nvcc -o matrix_multiply matrix_multiply.cu
hash_retrieval_kernel:
	nvcc -o ./hash_retrieval_kernel ./hash_retrieval_kernel.cu

