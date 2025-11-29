%%writefile helloCPUGPU.cu
#include <stdio.h>

__global__ void helloGPU() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU thread %d!\n", idx);
    printf("idx = %d\n", idx);
    printf("blockID = %d\n", blockIdx.x);
    printf("blockDim = %d\n", blockDim.x);
}

int main() {
    printf("Hello from CPU!\n");
    fflush(stdout);
    helloGPU<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
