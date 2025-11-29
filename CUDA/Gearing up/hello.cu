#include <iostream>
#include <cstdio>

// __global__ means this runs on GPU
__global__ void hello_from_gpu() {
    printf("Hello, World from GPU thread (%d, %d)!\n", threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "Hello, World from CPU!" << std::endl;

    // Launch 1 block with 5 threads
    hello_from_gpu<<<1, 5>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
