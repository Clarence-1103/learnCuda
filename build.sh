#!/usr/bin/bash
nvcc -o retrieval_kernel -I/usr/local/lib/python3.12/dist-packages/torch/include -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -I/usr/include/python3.12 ./retrieval_kernel.cu
