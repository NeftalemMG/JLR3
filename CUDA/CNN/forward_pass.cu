%%writefile forward_pass.cu

#ifndef FORWARD_PASS_CU
#define FORWARD_PASS_CU

#include "model.cu"

void forward_pass_mode0(CNNModel* model, const float* device_input_images, CudaTimer* timer) {
    NetworkConfig& config = model->config;
    
    int conv_output_size = config.batch_size * config.num_conv_filters * 
                          config.conv_output_height * config.conv_output_width;
    int pool_output_size = config.batch_size * config.num_conv_filters * 
                          config.pool_output_height * config.pool_output_width;
    
    if (timer) timer->start();
    
    int conv_blocks = (conv_output_size + config.threads_per_block - 1) / config.threads_per_block;
    conv2d_simple_kernel<<<conv_blocks, config.threads_per_block>>>(
        device_input_images, model->device_conv_weights, model->device_conv_bias, 
        model->device_conv_output,
        config.batch_size, config.num_input_channels, config.image_size, config.image_size,
        config.num_conv_filters, config.kernel_size, config.conv_padding, config.conv_stride, 
        config.conv_output_height, config.conv_output_width);
    
    relu_activation_kernel<<<conv_blocks, config.threads_per_block>>>(
        model->device_conv_output, model->device_activation_output, conv_output_size);
    
    int shared_memory_size = 2 * config.threads_per_block * sizeof(float);
    layernorm_kernel<<<config.batch_size, config.threads_per_block, shared_memory_size>>>(
        model->device_activation_output, model->device_normalized_output,
        model->device_layernorm_gamma, model->device_layernorm_beta,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, config.layernorm_epsilon);
    
    int pool_blocks = (pool_output_size + config.threads_per_block - 1) / config.threads_per_block;
    maxpool2x2_kernel<<<pool_blocks, config.threads_per_block>>>(
        model->device_normalized_output, model->device_pooled_output,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, 
        config.pool_output_height, config.pool_output_width);
    
    CHECK_CUDA(cudaMemcpy(model->device_fc_input, model->device_pooled_output,
                         pool_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int fc_output_size = config.batch_size * config.num_classes;
    int fc_blocks = (fc_output_size + config.threads_per_block - 1) / config.threads_per_block;
    fc_forward_kernel<<<fc_blocks, config.threads_per_block>>>(
        model->device_fc_input, model->device_fc_weights, model->device_fc_bias, 
        model->device_logits,
        config.batch_size, config.fc_input_dimension, config.num_classes);
    
    int softmax_blocks = (config.batch_size + config.threads_per_block - 1) / config.threads_per_block;
    softmax_kernel<<<softmax_blocks, config.threads_per_block>>>(
        model->device_logits, model->device_probabilities, 
        config.batch_size, config.num_classes);
    
    if (timer) {
        timer->stop();
        cudaDeviceSynchronize();
    }
}

void forward_pass_mode1(CNNModel* model, const float* device_input_images, CudaTimer* timer) {
    NetworkConfig& config = model->config;
    
    int conv_output_size = config.batch_size * config.num_conv_filters * 
                          config.conv_output_height * config.conv_output_width;
    int pool_output_size = config.batch_size * config.num_conv_filters * 
                          config.pool_output_height * config.pool_output_width;
    
    if (timer) timer->start();
    
    int conv_blocks = (conv_output_size + config.threads_per_block - 1) / config.threads_per_block;
    conv2d_simple_kernel<<<conv_blocks, config.threads_per_block>>>(
        device_input_images, model->device_conv_weights, model->device_conv_bias, 
        model->device_conv_output,
        config.batch_size, config.num_input_channels, config.image_size, config.image_size,
        config.num_conv_filters, config.kernel_size, config.conv_padding, config.conv_stride, 
        config.conv_output_height, config.conv_output_width);
    
    int shared_memory_size = 2 * config.threads_per_block * sizeof(float);
    gelu_layernorm_fused_kernel<<<config.batch_size, config.threads_per_block, shared_memory_size>>>(
        model->device_conv_output, model->device_normalized_output,
        model->device_layernorm_gamma, model->device_layernorm_beta,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, config.layernorm_epsilon);
    
    int pool_blocks = (pool_output_size + config.threads_per_block - 1) / config.threads_per_block;
    maxpool2x2_kernel<<<pool_blocks, config.threads_per_block>>>(
        model->device_normalized_output, model->device_pooled_output,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, 
        config.pool_output_height, config.pool_output_width);
    
    CHECK_CUDA(cudaMemcpy(model->device_fc_input, model->device_pooled_output,
                         pool_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int fc_output_size = config.batch_size * config.num_classes;
    int fc_blocks = (fc_output_size + config.threads_per_block - 1) / config.threads_per_block;
    fc_forward_kernel<<<fc_blocks, config.threads_per_block>>>(
        model->device_fc_input, model->device_fc_weights, model->device_fc_bias, 
        model->device_logits,
        config.batch_size, config.fc_input_dimension, config.num_classes);
    
    int softmax_blocks = (config.batch_size + config.threads_per_block - 1) / config.threads_per_block;
    softmax_kernel<<<softmax_blocks, config.threads_per_block>>>(
        model->device_logits, model->device_probabilities, 
        config.batch_size, config.num_classes);
    
    if (timer) {
        timer->stop();
        cudaDeviceSynchronize();
    }
}

void forward_pass_mode2(CNNModel* model, const float* device_input_images, CudaTimer* timer) {
    NetworkConfig& config = model->config;
    
    int conv_output_size = config.batch_size * config.num_conv_filters * 
                          config.conv_output_height * config.conv_output_width;
    int pool_output_size = config.batch_size * config.num_conv_filters * 
                          config.pool_output_height * config.pool_output_width;
    
    if (timer) timer->start();
    
    int conv_blocks = (conv_output_size + config.threads_per_block - 1) / config.threads_per_block;
    conv2d_simple_kernel<<<conv_blocks, config.threads_per_block>>>(
        device_input_images, model->device_conv_weights, model->device_conv_bias, 
        model->device_conv_output,
        config.batch_size, config.num_input_channels, config.image_size, config.image_size,
        config.num_conv_filters, config.kernel_size, config.conv_padding, config.conv_stride, 
        config.conv_output_height, config.conv_output_width);
    
    int evonorm_instances = config.batch_size * config.num_conv_filters;
    int evonorm_blocks = (evonorm_instances + config.threads_per_block - 1) / config.threads_per_block;
    evonorm_b0_kernel<<<evonorm_blocks, config.threads_per_block>>>(
        model->device_conv_output, model->device_normalized_output,
        model->device_evonorm_v1, model->device_evonorm_gamma,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, config.layernorm_epsilon);
    
    int pool_blocks = (pool_output_size + config.threads_per_block - 1) / config.threads_per_block;
    maxpool2x2_kernel<<<pool_blocks, config.threads_per_block>>>(
        model->device_normalized_output, model->device_pooled_output,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, 
        config.pool_output_height, config.pool_output_width);
    
    CHECK_CUDA(cudaMemcpy(model->device_fc_input, model->device_pooled_output,
                         pool_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int fc_output_size = config.batch_size * config.num_classes;
    int fc_blocks = (fc_output_size + config.threads_per_block - 1) / config.threads_per_block;
    fc_forward_kernel<<<fc_blocks, config.threads_per_block>>>(
        model->device_fc_input, model->device_fc_weights, model->device_fc_bias, 
        model->device_logits,
        config.batch_size, config.fc_input_dimension, config.num_classes);
    
    int softmax_blocks = (config.batch_size + config.threads_per_block - 1) / config.threads_per_block;
    softmax_kernel<<<softmax_blocks, config.threads_per_block>>>(
        model->device_logits, model->device_probabilities, 
        config.batch_size, config.num_classes);
    
    if (timer) {
        timer->stop();
        cudaDeviceSynchronize();
    }
}

void forward_pass_mode3(CNNModel* model, const float* device_input_images, CudaTimer* timer) {
    NetworkConfig& config = model->config;
    
    int conv_output_size = config.batch_size * config.num_conv_filters * 
                          config.conv_output_height * config.conv_output_width;
    int pool_output_size = config.batch_size * config.num_conv_filters * 
                          config.pool_output_height * config.pool_output_width;
    
    if (timer) timer->start();
    
    int conv_blocks = (conv_output_size + config.threads_per_block - 1) / config.threads_per_block;
    conv2d_simple_kernel<<<conv_blocks, config.threads_per_block>>>(
        device_input_images, model->device_conv_weights, model->device_conv_bias, 
        model->device_conv_output,
        config.batch_size, config.num_input_channels, config.image_size, config.image_size,
        config.num_conv_filters, config.kernel_size, config.conv_padding, config.conv_stride, 
        config.conv_output_height, config.conv_output_width);
    
    int shared_memory_size = 2 * config.threads_per_block * sizeof(float);
    swish_layernorm_fused_kernel<<<config.batch_size, config.threads_per_block, shared_memory_size>>>(
        model->device_conv_output, model->device_normalized_output,
        model->device_layernorm_gamma, model->device_layernorm_beta,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, 
        config.swish_beta, config.layernorm_epsilon);
    
    int pool_blocks = (pool_output_size + config.threads_per_block - 1) / config.threads_per_block;
    maxpool2x2_kernel<<<pool_blocks, config.threads_per_block>>>(
        model->device_normalized_output, model->device_pooled_output,
        config.batch_size, config.num_conv_filters, 
        config.conv_output_height, config.conv_output_width, 
        config.pool_output_height, config.pool_output_width);
    
    CHECK_CUDA(cudaMemcpy(model->device_fc_input, model->device_pooled_output,
                         pool_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int fc_output_size = config.batch_size * config.num_classes;
    int fc_blocks = (fc_output_size + config.threads_per_block - 1) / config.threads_per_block;
    fc_forward_kernel<<<fc_blocks, config.threads_per_block>>>(
        model->device_fc_input, model->device_fc_weights, model->device_fc_bias, 
        model->device_logits,
        config.batch_size, config.fc_input_dimension, config.num_classes);
    
    int softmax_blocks = (config.batch_size + config.threads_per_block - 1) / config.threads_per_block;
    softmax_kernel<<<softmax_blocks, config.threads_per_block>>>(
        model->device_logits, model->device_probabilities, 
        config.batch_size, config.num_classes);
    
    if (timer) {
        timer->stop();
        cudaDeviceSynchronize();
    }
}

void forward_pass(CNNModel* model, const float* device_input_images, int mode, CudaTimer* timer = nullptr) {
    switch(mode) {
        case 0:
            forward_pass_mode0(model, device_input_images, timer);
            break;
        case 1:
            forward_pass_mode1(model, device_input_images, timer);
            break;
        case 2:
            forward_pass_mode2(model, device_input_images, timer);
            break;
        case 3:
            forward_pass_mode3(model, device_input_images, timer);
            break;
        default:
            printf("Invalid mode: %d\n", mode);
            exit(1);
    }
}

#endif