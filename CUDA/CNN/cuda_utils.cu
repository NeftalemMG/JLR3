%%writefile cuda_utils.cu

#ifndef CUDA_UTILS_CU
#define CUDA_UTILS_CU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_milliseconds;
    
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        elapsed_milliseconds = 0.0f;
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_milliseconds, start_event, stop_event);
    }
    
    float get_elapsed_ms() const { 
        return elapsed_milliseconds; 
    }
};

float generate_random_float() {
    return ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

void initialize_random_weights(float* host_weights, int num_elements, float scale) {
    for (int i = 0; i < num_elements; i++) {
        host_weights[i] = generate_random_float() * scale;
    }
}

__global__ void zero_memory_kernel(float* data, int size) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= size) return;
    data[global_index] = 0.0f;
}

void zero_device_memory(float* device_ptr, int num_elements) {
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    zero_memory_kernel<<<num_blocks, threads_per_block>>>(device_ptr, num_elements);
}

#endif