%%writefile fused_kernels.cu

#ifndef FUSED_KERNELS_CU
#define FUSED_KERNELS_CU

#include <cuda_runtime.h>

__global__ void gelu_layernorm_fused_kernel(
    const float* input,
    float* output,
    const float* gamma_scale,
    const float* beta_shift,
    int batch_size,
    int num_channels,
    int spatial_height,
    int spatial_width,
    float epsilon
) {
    int sample_index = blockIdx.x;
    if (sample_index >= batch_size) return;
    
    int num_features = num_channels * spatial_height * spatial_width;
    int thread_index = threadIdx.x;
    
    extern __shared__ float shared_memory[];
    float* shared_sum = &shared_memory[0];
    float* shared_squared_sum = &shared_memory[blockDim.x];
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float gelu_coefficient = 0.044715f;
    
    float local_sum = 0.0f;
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        float input_value = input[global_index];
        
        float input_cubed = input_value * input_value * input_value;
        float inner_value = input_value + gelu_coefficient * input_cubed;
        float tanh_argument = sqrt_2_over_pi * inner_value;
        float gelu_output = 0.5f * input_value * (1.0f + tanhf(tanh_argument));
        
        local_sum += gelu_output;
    }
    shared_sum[thread_index] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
        }
        __syncthreads();
    }
    
    float mean_value = 0.0f;
    if (thread_index == 0) {
        mean_value = shared_sum[0] / num_features;
        shared_sum[0] = mean_value;
    }
    __syncthreads();
    mean_value = shared_sum[0];
    
    float local_squared_sum = 0.0f;
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        float input_value = input[global_index];
        
        float input_cubed = input_value * input_value * input_value;
        float inner_value = input_value + gelu_coefficient * input_cubed;
        float tanh_argument = sqrt_2_over_pi * inner_value;
        float gelu_output = 0.5f * input_value * (1.0f + tanhf(tanh_argument));
        
        float difference = gelu_output - mean_value;
        local_squared_sum += difference * difference;
    }
    shared_squared_sum[thread_index] = local_squared_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_squared_sum[thread_index] += shared_squared_sum[thread_index + stride];
        }
        __syncthreads();
    }
    
    float inverse_std = 0.0f;
    if (thread_index == 0) {
        float variance = shared_squared_sum[0] / num_features;
        inverse_std = rsqrtf(variance + epsilon);
        shared_squared_sum[0] = inverse_std;
    }
    __syncthreads();
    inverse_std = shared_squared_sum[0];
    
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        int channel_index = (feature_index / (spatial_height * spatial_width)) % num_channels;
        
        float input_value = input[global_index];
        
        float input_cubed = input_value * input_value * input_value;
        float inner_value = input_value + gelu_coefficient * input_cubed;
        float tanh_argument = sqrt_2_over_pi * inner_value;
        float gelu_output = 0.5f * input_value * (1.0f + tanhf(tanh_argument));
        
        float normalized_value = (gelu_output - mean_value) * inverse_std;
        output[global_index] = gamma_scale[channel_index] * normalized_value + beta_shift[channel_index];
    }
}

__global__ void swish_layernorm_fused_kernel(
    const float* input,
    float* output,
    const float* gamma_scale,
    const float* beta_shift,
    int batch_size,
    int num_channels,
    int spatial_height,
    int spatial_width,
    float swish_beta,
    float epsilon
) {
    int sample_index = blockIdx.x;
    if (sample_index >= batch_size) return;
    
    int num_features = num_channels * spatial_height * spatial_width;
    int thread_index = threadIdx.x;
    
    extern __shared__ float shared_memory[];
    float* shared_sum = &shared_memory[0];
    float* shared_squared_sum = &shared_memory[blockDim.x];
    
    float local_sum = 0.0f;
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        float input_value = input[global_index];
        
        float sigmoid_value = 1.0f / (1.0f + expf(-swish_beta * input_value));
        float swish_output = input_value * sigmoid_value;
        
        local_sum += swish_output;
    }
    shared_sum[thread_index] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
        }
        __syncthreads();
    }
    
    float mean_value = 0.0f;
    if (thread_index == 0) {
        mean_value = shared_sum[0] / num_features;
        shared_sum[0] = mean_value;
    }
    __syncthreads();
    mean_value = shared_sum[0];
    
    float local_squared_sum = 0.0f;
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        float input_value = input[global_index];
        
        float sigmoid_value = 1.0f / (1.0f + expf(-swish_beta * input_value));
        float swish_output = input_value * sigmoid_value;
        
        float difference = swish_output - mean_value;
        local_squared_sum += difference * difference;
    }
    shared_squared_sum[thread_index] = local_squared_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_squared_sum[thread_index] += shared_squared_sum[thread_index + stride];
        }
        __syncthreads();
    }
    
    float inverse_std = 0.0f;
    if (thread_index == 0) {
        float variance = shared_squared_sum[0] / num_features;
        inverse_std = rsqrtf(variance + epsilon);
        shared_squared_sum[0] = inverse_std;
    }
    __syncthreads();
    inverse_std = shared_squared_sum[0];
    
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        int channel_index = (feature_index / (spatial_height * spatial_width)) % num_channels;
        
        float input_value = input[global_index];
        
        float sigmoid_value = 1.0f / (1.0f + expf(-swish_beta * input_value));
        float swish_output = input_value * sigmoid_value;
        
        float normalized_value = (swish_output - mean_value) * inverse_std;
        output[global_index] = gamma_scale[channel_index] * normalized_value + beta_shift[channel_index];
    }
}

#endif