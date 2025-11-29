%%writefile normalization_kernels.cu

#ifndef NORMALIZATION_KERNELS_CU
#define NORMALIZATION_KERNELS_CU

#include <cuda_runtime.h>

__global__ void layernorm_kernel(
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
    
    float local_sum = 0.0f;
    for (int feature_index = thread_index; feature_index < num_features; feature_index += blockDim.x) {
        int global_index = sample_index * num_features + feature_index;
        local_sum += input[global_index];
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
        float difference = input[global_index] - mean_value;
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
        float normalized_value = (input[global_index] - mean_value) * inverse_std;
        output[global_index] = gamma_scale[channel_index] * normalized_value + beta_shift[channel_index];
    }
}

__global__ void evonorm_b0_kernel(
    const float* input,
    float* output,
    const float* v1_parameter,
    const float* gamma_offset,
    int batch_size,
    int num_channels,
    int spatial_height,
    int spatial_width,
    float epsilon
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_instances = batch_size * num_channels;
    
    if (global_index >= total_instances) return;
    
    int channel_index = global_index % num_channels;
    int sample_index = global_index / num_channels;
    int spatial_size = spatial_height * spatial_width;
    
    float sum_of_squares = 0.0f;
    for (int spatial_index = 0; spatial_index < spatial_size; spatial_index++) {
        int element_index = ((sample_index * num_channels + channel_index) * spatial_height * spatial_width) + spatial_index;
        float input_value = input[element_index];
        sum_of_squares += input_value * input_value;
    }
    
    float instance_std = sqrtf(sum_of_squares / spatial_size + epsilon);
    float denominator = fmaxf(v1_parameter[channel_index] * instance_std, epsilon);
    
    for (int spatial_index = 0; spatial_index < spatial_size; spatial_index++) {
        int element_index = ((sample_index * num_channels + channel_index) * spatial_height * spatial_width) + spatial_index;
        float input_value = input[element_index];
        output[element_index] = (input_value / denominator) + gamma_offset[channel_index];
    }
}

__global__ void init_layernorm_params_kernel(
    float* gamma_scale, 
    float* beta_shift, 
    int num_channels
) {
    int channel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel_index >= num_channels) return;
    
    gamma_scale[channel_index] = 1.0f;
    beta_shift[channel_index] = 0.0f;
}

__global__ void init_evonorm_params_kernel(
    float* v1_parameter, 
    float* gamma_offset, 
    int num_channels
) {
    int channel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel_index >= num_channels) return;
    
    v1_parameter[channel_index] = 1.0f;
    gamma_offset[channel_index] = 0.0f;
}

#endif