%%writefile model.cu

#ifndef MODEL_CU
#define MODEL_CU

#include <cuda_runtime.h>
#include "cuda_utils.cu"
#include "activation_kernels.cu"
#include "normalization_kernels.cu"
#include "fused_kernels.cu"
#include "basic_kernels.cu"
#include "training_kernels.cu"

struct NetworkConfig {
    int num_classes;
    int image_size;
    int num_input_channels;
    int num_conv_filters;
    int kernel_size;
    int conv_padding;
    int conv_stride;
    int conv_output_height;
    int conv_output_width;
    int pool_output_height;
    int pool_output_width;
    int fc_input_dimension;
    int batch_size;
    int num_epochs;
    float learning_rate;
    int threads_per_block;
    float layernorm_epsilon;
    float swish_beta;
};

NetworkConfig create_default_config() {
    NetworkConfig config;
    
    config.num_classes = 10;
    config.image_size = 28;
    config.num_input_channels = 1;
    config.num_conv_filters = 16;
    config.kernel_size = 3;
    config.conv_padding = 1;
    config.conv_stride = 1;
    config.conv_output_height = 28;
    config.conv_output_width = 28;
    config.pool_output_height = 14;
    config.pool_output_width = 14;
    config.fc_input_dimension = config.num_conv_filters * 
                                config.pool_output_height * 
                                config.pool_output_width;
    config.batch_size = 64;
    config.num_epochs = 3;
    config.learning_rate = 0.01f;
    config.threads_per_block = 256;
    config.layernorm_epsilon = 1e-5f;
    config.swish_beta = 1.0f;
    
    return config;
}

struct CNNModel {
    NetworkConfig config;
    
    float* device_conv_weights;
    float* device_conv_bias;
    float* device_fc_weights;
    float* device_fc_bias;
    float* device_layernorm_gamma;
    float* device_layernorm_beta;
    float* device_evonorm_v1;
    float* device_evonorm_gamma;
    float* device_gradient_fc_weights;
    float* device_gradient_fc_bias;
    float* device_conv_output;
    float* device_activation_output;
    float* device_normalized_output;
    float* device_pooled_output;
    float* device_fc_input;
    float* device_logits;
    float* device_probabilities;
    float* device_gradient_fc_input;
    float* device_gradient_logits;
};

CNNModel* allocate_model(NetworkConfig config) {
    CNNModel* model = new CNNModel();
    model->config = config;
    
    int conv_output_size = config.batch_size * config.num_conv_filters * 
                          config.conv_output_height * config.conv_output_width;
    int pool_output_size = config.batch_size * config.num_conv_filters * 
                          config.pool_output_height * config.pool_output_width;
    
    CHECK_CUDA(cudaMalloc(&model->device_conv_weights, 
               config.num_conv_filters * config.num_input_channels * 
               config.kernel_size * config.kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_conv_bias, 
               config.num_conv_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_fc_weights, 
               config.num_classes * config.fc_input_dimension * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_fc_bias, 
               config.num_classes * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_layernorm_gamma, 
               config.num_conv_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_layernorm_beta, 
               config.num_conv_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_evonorm_v1, 
               config.num_conv_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_evonorm_gamma, 
               config.num_conv_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_gradient_fc_weights, 
               config.num_classes * config.fc_input_dimension * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_gradient_fc_bias, 
               config.num_classes * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_conv_output, 
               conv_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_activation_output, 
               conv_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_normalized_output, 
               conv_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_pooled_output, 
               pool_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_fc_input, 
               config.batch_size * config.fc_input_dimension * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_logits, 
               config.batch_size * config.num_classes * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_probabilities, 
               config.batch_size * config.num_classes * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_gradient_fc_input, 
               config.batch_size * config.fc_input_dimension * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->device_gradient_logits, 
               config.batch_size * config.num_classes * sizeof(float)));
    
    return model;
}

void initialize_model(CNNModel* model) {
    srand(42);
    
    NetworkConfig& config = model->config;
    
    int conv_weight_count = config.num_conv_filters * config.num_input_channels * 
                           config.kernel_size * config.kernel_size;
    float* host_conv_weights = (float*)malloc(conv_weight_count * sizeof(float));
    float conv_scale = sqrtf(2.0f / (config.num_input_channels * 
                                    config.kernel_size * config.kernel_size));
    initialize_random_weights(host_conv_weights, conv_weight_count, conv_scale);
    CHECK_CUDA(cudaMemcpy(model->device_conv_weights, host_conv_weights, 
                         conv_weight_count * sizeof(float), cudaMemcpyHostToDevice));
    free(host_conv_weights);
    
    CHECK_CUDA(cudaMemset(model->device_conv_bias, 0, 
                         config.num_conv_filters * sizeof(float)));
    
    int param_blocks = (config.num_conv_filters + 255) / 256;
    init_layernorm_params_kernel<<<param_blocks, 256>>>(
        model->device_layernorm_gamma, 
        model->device_layernorm_beta, 
        config.num_conv_filters);
    
    init_evonorm_params_kernel<<<param_blocks, 256>>>(
        model->device_evonorm_v1, 
        model->device_evonorm_gamma, 
        config.num_conv_filters);
    
    int fc_weight_count = config.num_classes * config.fc_input_dimension;
    float* host_fc_weights = (float*)malloc(fc_weight_count * sizeof(float));
    float fc_scale = sqrtf(2.0f / config.fc_input_dimension);
    initialize_random_weights(host_fc_weights, fc_weight_count, fc_scale);
    CHECK_CUDA(cudaMemcpy(model->device_fc_weights, host_fc_weights, 
                         fc_weight_count * sizeof(float), cudaMemcpyHostToDevice));
    free(host_fc_weights);
    
    CHECK_CUDA(cudaMemset(model->device_fc_bias, 0, 
                         config.num_classes * sizeof(float)));
    
    cudaDeviceSynchronize();
}

void free_model(CNNModel* model) {
    cudaFree(model->device_conv_weights);
    cudaFree(model->device_conv_bias);
    cudaFree(model->device_fc_weights);
    cudaFree(model->device_fc_bias);
    cudaFree(model->device_layernorm_gamma);
    cudaFree(model->device_layernorm_beta);
    cudaFree(model->device_evonorm_v1);
    cudaFree(model->device_evonorm_gamma);
    cudaFree(model->device_gradient_fc_weights);
    cudaFree(model->device_gradient_fc_bias);
    cudaFree(model->device_conv_output);
    cudaFree(model->device_activation_output);
    cudaFree(model->device_normalized_output);
    cudaFree(model->device_pooled_output);
    cudaFree(model->device_fc_input);
    cudaFree(model->device_logits);
    cudaFree(model->device_probabilities);
    cudaFree(model->device_gradient_fc_input);
    cudaFree(model->device_gradient_logits);
    delete model;
}

#endif