%%writefile activation_kernels.cu

#ifndef ACTIVATION_KERNELS_CU
#define ACTIVATION_KERNELS_CU

#include <cuda_runtime.h>

__global__ void relu_activation_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= total_elements) return;
    
    float input_value = input[global_index];
    output[global_index] = fmaxf(0.0f, input_value);
}

__global__ void gelu_activation_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= total_elements) return;
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float gelu_coefficient = 0.044715f;
    
    float input_value = input[global_index];
    float input_cubed = input_value * input_value * input_value;
    float inner_value = input_value + gelu_coefficient * input_cubed;
    float tanh_argument = sqrt_2_over_pi * inner_value;
    float gelu_output = 0.5f * input_value * (1.0f + tanhf(tanh_argument));
    
    output[global_index] = gelu_output;
}

__global__ void swish_activation_kernel(
    const float* input,
    float* output,
    int total_elements,
    float swish_beta
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= total_elements) return;
    
    float input_value = input[global_index];
    float sigmoid_value = 1.0f / (1.0f + expf(-swish_beta * input_value));
    float swish_output = input_value * sigmoid_value;
    
    output[global_index] = swish_output;
}

#endif