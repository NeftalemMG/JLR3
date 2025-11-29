%%writefile training_kernels.cu

#ifndef TRAINING_KERNELS_CU
#define TRAINING_KERNELS_CU

#include <cuda_runtime.h>

__global__ void softmax_cross_entropy_backward_kernel(
    const float* probabilities,
    const unsigned char* labels,
    float* gradient_logits,
    int batch_size,
    int num_classes
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_classes;
    
    if (global_index >= total_elements) return;
    
    int sample_index = global_index / num_classes;
    int class_index = global_index % num_classes;
    int true_label = labels[sample_index];
    
    float gradient = probabilities[global_index];
    if (class_index == true_label) {
        gradient -= 1.0f;
    }
    
    gradient_logits[global_index] = gradient / batch_size;
}

__global__ void fc_backward_kernel(
    const float* gradient_output,
    const float* input,
    const float* weights,
    float* gradient_input,
    float* gradient_weights,
    float* gradient_bias,
    int batch_size,
    int input_dimension,
    int output_dimension
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_input_elements = batch_size * input_dimension;
    if (global_index < total_input_elements) {
        int sample_index = global_index / input_dimension;
        int input_index = global_index % input_dimension;
        
        float gradient_sum = 0.0f;
        for (int output_index = 0; output_index < output_dimension; output_index++) {
            float grad_out = gradient_output[sample_index * output_dimension + output_index];
            float weight = weights[output_index * input_dimension + input_index];
            gradient_sum += grad_out * weight;
        }
        gradient_input[global_index] = gradient_sum;
    }
    
    int total_weight_elements = output_dimension * input_dimension;
    if (global_index < total_weight_elements) {
        int output_index = global_index / input_dimension;
        int input_index = global_index % input_dimension;
        
        float gradient_weight = 0.0f;
        for (int sample_index = 0; sample_index < batch_size; sample_index++) {
            float grad_out = gradient_output[sample_index * output_dimension + output_index];
            float input_val = input[sample_index * input_dimension + input_index];
            gradient_weight += grad_out * input_val;
        }
        
        atomicAdd(&gradient_weights[global_index], gradient_weight);
    }
    
    if (global_index < output_dimension) {
        float gradient_bias_sum = 0.0f;
        for (int sample_index = 0; sample_index < batch_size; sample_index++) {
            gradient_bias_sum += gradient_output[sample_index * output_dimension + global_index];
        }
        atomicAdd(&gradient_bias[global_index], gradient_bias_sum);
    }
}

__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    int num_elements,
    float learning_rate
) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= num_elements) return;
    
    weights[global_index] -= learning_rate * gradients[global_index];
}

__global__ void compute_accuracy_kernel(
    const float* probabilities,
    const unsigned char* labels,
    int* correct_count,
    int batch_size,
    int num_classes
) {
    __shared__ int shared_correct[256];
    
    int thread_index = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + thread_index;
    
    int local_correct = 0;
    if (global_index < batch_size) {
        int base_index = global_index * num_classes;
        
        int predicted_class = 0;
        float max_probability = probabilities[base_index];
        
        for (int class_index = 1; class_index < num_classes; class_index++) {
            float probability = probabilities[base_index + class_index];
            if (probability > max_probability) {
                max_probability = probability;
                predicted_class = class_index;
            }
        }
        
        if (predicted_class == labels[global_index]) {
            local_correct = 1;
        }
    }
    
    shared_correct[thread_index] = local_correct;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_correct[thread_index] += shared_correct[thread_index + stride];
        }
        __syncthreads();
    }
    
    if (thread_index == 0) {
        atomicAdd(correct_count, shared_correct[0]);
    }
}

#endif