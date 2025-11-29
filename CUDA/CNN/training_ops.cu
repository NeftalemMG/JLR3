%%writefile training_ops.cu

#ifndef TRAINING_OPS_CU
#define TRAINING_OPS_CU

#include "forward_pass.cu"

void train_step(CNNModel* model, const float* device_batch_images, 
                const unsigned char* device_batch_labels, int mode) {
    NetworkConfig& config = model->config;
    
    int fc_gradient_size = config.num_classes * config.fc_input_dimension;
    zero_device_memory(model->device_gradient_fc_weights, fc_gradient_size);
    zero_device_memory(model->device_gradient_fc_bias, config.num_classes);
    
    forward_pass(model, device_batch_images, mode, nullptr);
    
    int fc_output_size = config.batch_size * config.num_classes;
    int logits_blocks = (fc_output_size + config.threads_per_block - 1) / config.threads_per_block;
    softmax_cross_entropy_backward_kernel<<<logits_blocks, config.threads_per_block>>>(
        model->device_probabilities, device_batch_labels, model->device_gradient_logits, 
        config.batch_size, config.num_classes);
    
    int max_fc_elements = max(config.batch_size * config.fc_input_dimension, 
                             max(config.num_classes * config.fc_input_dimension, 
                                 config.num_classes));
    int fc_backward_blocks = (max_fc_elements + config.threads_per_block - 1) / config.threads_per_block;
    fc_backward_kernel<<<fc_backward_blocks, config.threads_per_block>>>(
        model->device_gradient_logits, model->device_fc_input, model->device_fc_weights,
        model->device_gradient_fc_input, model->device_gradient_fc_weights, 
        model->device_gradient_fc_bias,
        config.batch_size, config.fc_input_dimension, config.num_classes);
    
    sgd_update_kernel<<<(fc_gradient_size + 255) / 256, 256>>>(
        model->device_fc_weights, model->device_gradient_fc_weights, 
        fc_gradient_size, config.learning_rate);
    sgd_update_kernel<<<(config.num_classes + 255) / 256, 256>>>(
        model->device_fc_bias, model->device_gradient_fc_bias, 
        config.num_classes, config.learning_rate);
    
    cudaDeviceSynchronize();
}

float evaluate_model(CNNModel* model, const float* device_test_images, 
                     const unsigned char* device_test_labels, 
                     int num_test_samples, int mode, int* device_correct_count) {
    NetworkConfig& config = model->config;
    int num_batches = num_test_samples / config.batch_size;
    int total_correct = 0;
    
    for (int batch_index = 0; batch_index < num_batches; batch_index++) {
        int batch_start = batch_index * config.batch_size;
        const float* device_batch_images = device_test_images + 
                                           batch_start * config.image_size * config.image_size;
        const unsigned char* device_batch_labels = device_test_labels + batch_start;
        
        forward_pass(model, device_batch_images, mode, nullptr);
        
        CHECK_CUDA(cudaMemset(device_correct_count, 0, sizeof(int)));
        
        compute_accuracy_kernel<<<(config.batch_size + 255) / 256, 256>>>(
            model->device_probabilities, device_batch_labels, device_correct_count, 
            config.batch_size, config.num_classes);
        
        int batch_correct;
        CHECK_CUDA(cudaMemcpy(&batch_correct, device_correct_count, 
                             sizeof(int), cudaMemcpyDeviceToHost));
        total_correct += batch_correct;
    }
    
    return 100.0f * total_correct / (num_batches * config.batch_size);
}

#endif