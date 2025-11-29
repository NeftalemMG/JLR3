%%writefile basic_kernels.cu

#ifndef BASIC_KERNELS_CU
#define BASIC_KERNELS_CU

#include <cuda_runtime.h>

__global__ void conv2d_simple_kernel(
    const float* input,
    const float* conv_weights,
    const float* conv_bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_size,
    int padding,
    int stride,
    int output_height,
    int output_width
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_channels * output_height * output_width;
    
    if (global_index >= total_elements) return;
    
    int temp_index = global_index;
    int out_x = temp_index % output_width;
    temp_index /= output_width;
    int out_y = temp_index % output_height;
    temp_index /= output_height;
    int out_channel = temp_index % output_channels;
    temp_index /= output_channels;
    int sample_index = temp_index;
    
    float accumulator = 0.0f;
    for (int in_channel = 0; in_channel < input_channels; in_channel++) {
        for (int kernel_y = 0; kernel_y < kernel_size; kernel_y++) {
            for (int kernel_x = 0; kernel_x < kernel_size; kernel_x++) {
                int input_y = out_y * stride + kernel_y - padding;
                int input_x = out_x * stride + kernel_x - padding;
                
                if (input_y >= 0 && input_y < input_height && 
                    input_x >= 0 && input_x < input_width) {
                    
                    int input_index = ((sample_index * input_channels + in_channel) * input_height + input_y) * input_width + input_x;
                    int weight_index = ((out_channel * input_channels + in_channel) * kernel_size + kernel_y) * kernel_size + kernel_x;
                    
                    accumulator += input[input_index] * conv_weights[weight_index];
                }
            }
        }
    }
    
    if (conv_bias) {
        accumulator += conv_bias[out_channel];
    }
    
    output[global_index] = accumulator;
}

__global__ void maxpool2x2_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * output_height * output_width;
    
    if (global_index >= total_elements) return;
    
    int temp_index = global_index;
    int out_x = temp_index % output_width;
    temp_index /= output_width;
    int out_y = temp_index % output_height;
    temp_index /= output_height;
    int channel = temp_index % num_channels;
    temp_index /= num_channels;
    int sample_index = temp_index;
    
    int input_y_start = out_y * 2;
    int input_x_start = out_x * 2;
    
    float max_value = -1e30f;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int input_y = input_y_start + dy;
            int input_x = input_x_start + dx;
            
            if (input_y < input_height && input_x < input_width) {
                int input_index = ((sample_index * num_channels + channel) * input_height + input_y) * input_width + input_x;
                float value = input[input_index];
                if (value > max_value) {
                    max_value = value;
                }
            }
        }
    }
    
    output[global_index] = max_value;
}

__global__ void fc_forward_kernel(
    const float* input,
    const float* fc_weights,
    const float* fc_bias,
    float* output,
    int batch_size,
    int input_dimension,
    int output_dimension
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_dimension;
    
    if (global_index >= total_elements) return;
    
    int output_index = global_index % output_dimension;
    int sample_index = global_index / output_dimension;
    
    float accumulator = 0.0f;
    for (int i = 0; i < input_dimension; i++) {
        accumulator += input[sample_index * input_dimension + i] * 
                       fc_weights[output_index * input_dimension + i];
    }
    
    if (fc_bias) {
        accumulator += fc_bias[output_index];
    }
    
    output[global_index] = accumulator;
}

__global__ void softmax_kernel(
    const float* logits,
    float* probabilities,
    int batch_size,
    int num_classes
) {
    int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= batch_size) return;
    
    int base_index = sample_index * num_classes;
    
    float max_logit = logits[base_index];
    for (int class_index = 1; class_index < num_classes; class_index++) {
        float logit_value = logits[base_index + class_index];
        if (logit_value > max_logit) {
            max_logit = logit_value;
        }
    }
    
    float sum_exp = 0.0f;
    for (int class_index = 0; class_index < num_classes; class_index++) {
        float exp_value = expf(logits[base_index + class_index] - max_logit);
        probabilities[base_index + class_index] = exp_value;
        sum_exp += exp_value;
    }
    
    for (int class_index = 0; class_index < num_classes; class_index++) {
        probabilities[base_index + class_index] /= sum_exp;
    }
}

#endif