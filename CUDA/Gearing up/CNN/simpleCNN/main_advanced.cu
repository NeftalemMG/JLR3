// ============================================================================
// MAIN_ADVANCED.CU - CNN WITH ADVANCED COMPONENTS & KERNEL FUSION
// ============================================================================
// This is the main entry point for testing:
// 1. Non-fused implementations (baseline)
// 2. Four different kernel fusion techniques
// 3. Performance profiling and comparison
//
// ARCHITECTURE:
// Input (MNIST) → Conv2D → [Activation + Normalization] → MaxPool → FC → Loss
//
// The [Activation + Normalization] part can be:
// - Non-fused: Separate GELU/Swish → LayerNorm kernels
// - Fused V1: GELU + LayerNorm in single kernel
// - Fused V2: EvoNorm-B0
// - Fused V3: Swish + LayerNorm
// - Fused V4: Conv + GELU + LayerNorm
//
// COMPILATION:
// nvcc -o mnist_advanced main_advanced.cu -O3 -arch=sm_75
//
// USAGE:
// ./mnist_advanced [mode]
// where mode = 0 (non-fused), 1 (V1), 2 (V2), 3 (V3), 4 (V4), 5 (compare all)
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include all our custom modules
#include "mnist_loader.cu"      // MNIST dataset loading
#include "kernels.cu"           // Basic kernels (conv, maxpool, fc)
#include "utils.cu"             // Utilities and error checking
#include "advanced_kernels.cu"  // Non-fused advanced kernels
#include "fused_kernels.cu"     // Fused kernel implementations
#include "profiling_utils.cu"   // Performance measurement utilities

// ============================================================================
// GLOBAL CONFIGURATION
// ============================================================================
// These parameters define our CNN architecture and training setup
// ============================================================================

// Architecture hyperparameters
#define NUM_CLASSES 10          // MNIST has 10 digits (0-9)
#define IMAGE_SIZE 28           // MNIST images are 28x28
#define NUM_INPUT_CHANNELS 1    // Grayscale images

// Conv layer configuration
#define NUM_CONV_FILTERS 32     // Number of output channels
#define KERNEL_SIZE 3           // 3x3 convolution
#define CONV_PADDING 1          // Same padding
#define CONV_STRIDE 1           // Stride of 1

// Compute output dimensions
#define CONV_OUT_H ((IMAGE_SIZE + 2 * CONV_PADDING - KERNEL_SIZE) / CONV_STRIDE + 1)
#define CONV_OUT_W ((IMAGE_SIZE + 2 * CONV_PADDING - KERNEL_SIZE) / CONV_STRIDE + 1)
#define POOL_OUT_H (CONV_OUT_H / 2)
#define POOL_OUT_W (CONV_OUT_W / 2)
#define FC_INPUT_DIM (NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W)

// Training configuration
#define BATCH_SIZE 64           // Number of samples per batch
#define NUM_EPOCHS 3            // Number of training epochs
#define LEARNING_RATE 0.001f    // Learning rate (not used in forward-only pass)

// Kernel launch configuration
#define THREADS_PER_BLOCK 256   // Standard choice for modern GPUs

// LayerNorm epsilon
#define LAYERNORM_EPS 1e-5f

// Swish beta parameter
#define SWISH_BETA 1.0f

// Focal loss parameters
#define FOCAL_ALPHA 0.25f
#define FOCAL_GAMMA 2.0f

// Label smoothing parameter
#define LABEL_SMOOTHING 0.1f

// ============================================================================
// MODEL PARAMETERS STRUCTURE
// ============================================================================
// This structure holds all learnable parameters for the model
// ============================================================================

typedef struct {
    // Convolution layer
    float* d_conv_weights;     // [NUM_CONV_FILTERS, NUM_INPUT_CHANNELS, K, K]
    float* d_conv_bias;        // [NUM_CONV_FILTERS]
    
    // LayerNorm parameters (used in non-fused and V1, V3)
    float* d_ln_gamma;         // [NUM_CONV_FILTERS]
    float* d_ln_beta;          // [NUM_CONV_FILTERS]
    
    // EvoNorm parameters (used in V2)
    float* d_evonorm_v1;       // [NUM_CONV_FILTERS]
    float* d_evonorm_gamma;    // [NUM_CONV_FILTERS]
    
    // Fully connected layer
    float* d_fc_weights;       // [NUM_CLASSES, FC_INPUT_DIM]
    float* d_fc_bias;          // [NUM_CLASSES]
    
    // Intermediate buffers
    float* d_conv_out;         // After convolution
    float* d_activated;        // After activation (non-fused only)
    float* d_normalized;       // After normalization
    float* d_pooled;           // After max pooling
    float* d_fc_in;            // Flattened pool output
    float* d_logits;           // Final output logits
    
} ModelParams;

// ============================================================================
// FORWARD PASS IMPLEMENTATIONS
// ============================================================================
// Different forward pass implementations for each fusion mode
// ============================================================================

// ----------------------------------------------------------------------------
// MODE 0: NON-FUSED (BASELINE)
// ----------------------------------------------------------------------------
// This is our baseline implementation where each operation is a separate kernel
// Conv → GELU → LayerNorm → MaxPool → FC
//
// This gives us the reference accuracy and baseline performance
// ----------------------------------------------------------------------------

void forward_pass_non_fused(
    ModelParams* model,
    const float* d_input,
    const unsigned char* labels,
    int batch_size,
    KernelStats* stats_array,
    int* stats_idx
) {
    int conv_out_size = batch_size * NUM_CONV_FILTERS * CONV_OUT_H * CONV_OUT_W;
    int pool_out_size = batch_size * NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
    
    // ========================================================================
    // STEP 1: Convolution
    // ========================================================================
    int conv_blocks = (conv_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    CudaTimer timer;
    timer.start();
    conv2d_simple_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(
        d_input,
        model->d_conv_weights,
        model->d_conv_bias,
        model->d_conv_out,
        batch_size,
        NUM_INPUT_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CONV_FILTERS,
        KERNEL_SIZE,
        CONV_PADDING,
        CONV_STRIDE,
        CONV_OUT_H,
        CONV_OUT_W
    );
    timer.stop();
    
    // Record stats
    KernelStats conv_stats("Conv2D");
    conv_stats.execution_time_ms = timer.get_elapsed_ms();
    conv_stats.memory_bytes_read = batch_size * IMAGE_SIZE * IMAGE_SIZE * sizeof(float) +
                                    NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    conv_stats.memory_bytes_written = conv_out_size * sizeof(float);
    conv_stats.blocks = conv_blocks;
    conv_stats.threads_per_block = THREADS_PER_BLOCK;
    conv_stats.total_threads = conv_out_size;
    conv_stats.compute_metrics();
    stats_array[(*stats_idx)++] = conv_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 2: GELU Activation
    // ========================================================================
    timer.start();
    gelu_activation_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(
        model->d_conv_out,
        model->d_activated,
        conv_out_size
    );
    timer.stop();
    
    KernelStats gelu_stats("GELU Activation");
    gelu_stats.execution_time_ms = timer.get_elapsed_ms();
    gelu_stats.memory_bytes_read = conv_out_size * sizeof(float);
    gelu_stats.memory_bytes_written = conv_out_size * sizeof(float);
    gelu_stats.blocks = conv_blocks;
    gelu_stats.threads_per_block = THREADS_PER_BLOCK;
    gelu_stats.total_threads = conv_out_size;
    gelu_stats.compute_metrics();
    stats_array[(*stats_idx)++] = gelu_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 3: Layer Normalization
    // ========================================================================
    int ln_blocks = batch_size;  // One block per sample
    int shared_mem_size = 2 * THREADS_PER_BLOCK * sizeof(float);
    
    timer.start();
    layernorm_forward_kernel<<<ln_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        model->d_activated,
        model->d_normalized,
        model->d_ln_gamma,
        model->d_ln_beta,
        batch_size,
        NUM_CONV_FILTERS,
        CONV_OUT_H,
        CONV_OUT_W,
        LAYERNORM_EPS
    );
    timer.stop();
    
    KernelStats ln_stats("LayerNorm");
    ln_stats.execution_time_ms = timer.get_elapsed_ms();
    ln_stats.memory_bytes_read = conv_out_size * sizeof(float) + 
                                  2 * NUM_CONV_FILTERS * sizeof(float);
    ln_stats.memory_bytes_written = conv_out_size * sizeof(float);
    ln_stats.blocks = ln_blocks;
    ln_stats.threads_per_block = THREADS_PER_BLOCK;
    ln_stats.total_threads = batch_size * THREADS_PER_BLOCK;
    ln_stats.compute_metrics();
    stats_array[(*stats_idx)++] = ln_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 4: MaxPool 2x2
    // ========================================================================
    int pool_blocks = (pool_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    maxpool2x2_kernel<<<pool_blocks, THREADS_PER_BLOCK>>>(
        model->d_normalized,
        model->d_pooled,
        batch_size,
        NUM_CONV_FILTERS,
        CONV_OUT_H,
        CONV_OUT_W,
        POOL_OUT_H,
        POOL_OUT_W
    );
    timer.stop();
    
    KernelStats pool_stats("MaxPool2x2");
    pool_stats.execution_time_ms = timer.get_elapsed_ms();
    pool_stats.memory_bytes_read = conv_out_size * sizeof(float);
    pool_stats.memory_bytes_written = pool_out_size * sizeof(float);
    pool_stats.blocks = pool_blocks;
    pool_stats.threads_per_block = THREADS_PER_BLOCK;
    pool_stats.total_threads = pool_out_size;
    pool_stats.compute_metrics();
    stats_array[(*stats_idx)++] = pool_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 5: Flatten (implicit - just pointer manipulation)
    // ========================================================================
    CHECK_CUDA(cudaMemcpy(model->d_fc_in, model->d_pooled, 
                         pool_out_size * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // ========================================================================
    // STEP 6: Fully Connected Layer
    // ========================================================================
    int fc_out_size = batch_size * NUM_CLASSES;
    int fc_blocks = (fc_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    fc_forward_kernel<<<fc_blocks, THREADS_PER_BLOCK>>>(
        model->d_fc_in,
        model->d_fc_weights,
        model->d_fc_bias,
        model->d_logits,
        batch_size,
        FC_INPUT_DIM,
        NUM_CLASSES
    );
    timer.stop();
    
    KernelStats fc_stats("Fully Connected");
    fc_stats.execution_time_ms = timer.get_elapsed_ms();
    fc_stats.memory_bytes_read = pool_out_size * sizeof(float) + 
                                  NUM_CLASSES * FC_INPUT_DIM * sizeof(float);
    fc_stats.memory_bytes_written = fc_out_size * sizeof(float);
    fc_stats.blocks = fc_blocks;
    fc_stats.threads_per_block = THREADS_PER_BLOCK;
    fc_stats.total_threads = fc_out_size;
    fc_stats.compute_metrics();
    stats_array[(*stats_idx)++] = fc_stats;
    
    cudaDeviceSynchronize();
}



// ----------------------------------------------------------------------------
// MODE 1: FUSED V1 (GELU + LAYERNORM)
// ----------------------------------------------------------------------------
// Combines GELU activation and LayerNorm into a single kernel
// Conv → [GELU+LayerNorm] → MaxPool → FC
//
// Expected benefit: ~1.5x speedup on activation+normalization stage
// Memory savings: 1 global write + 1 global read eliminated
// ----------------------------------------------------------------------------

void forward_pass_fused_v1(
    ModelParams* model,
    const float* d_input,
    const unsigned char* labels,
    int batch_size,
    KernelStats* stats_array,
    int* stats_idx
) {
    int conv_out_size = batch_size * NUM_CONV_FILTERS * CONV_OUT_H * CONV_OUT_W;
    int pool_out_size = batch_size * NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
    
    CudaTimer timer;
    
    // ========================================================================
    // STEP 1: Convolution (same as non-fused)
    // ========================================================================
    int conv_blocks = (conv_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    conv2d_simple_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(
        d_input, model->d_conv_weights, model->d_conv_bias, model->d_conv_out,
        batch_size, NUM_INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
        NUM_CONV_FILTERS, KERNEL_SIZE, CONV_PADDING, CONV_STRIDE,
        CONV_OUT_H, CONV_OUT_W
    );
    timer.stop();
    
    KernelStats conv_stats("Conv2D [V1]");
    conv_stats.execution_time_ms = timer.get_elapsed_ms();
    conv_stats.memory_bytes_read = batch_size * IMAGE_SIZE * IMAGE_SIZE * sizeof(float) +
                                    NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    conv_stats.memory_bytes_written = conv_out_size * sizeof(float);
    conv_stats.compute_metrics();
    stats_array[(*stats_idx)++] = conv_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 2: FUSED GELU + LayerNorm
    // ========================================================================
    int fused_blocks = batch_size;
    int shared_mem_size = 2 * THREADS_PER_BLOCK * sizeof(float);
    
    timer.start();
    gelu_layernorm_fused_kernel<<<fused_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        model->d_conv_out,
        model->d_normalized,
        model->d_ln_gamma,
        model->d_ln_beta,
        batch_size,
        NUM_CONV_FILTERS,
        CONV_OUT_H,
        CONV_OUT_W,
        LAYERNORM_EPS
    );
    timer.stop();
    
    KernelStats fused_stats("GELU+LayerNorm FUSED [V1]");
    fused_stats.execution_time_ms = timer.get_elapsed_ms();
    // Memory: Read input + gamma + beta, Write output
    // Note: GELU intermediate result NOT written to memory!
    fused_stats.memory_bytes_read = conv_out_size * sizeof(float) + 
                                      2 * NUM_CONV_FILTERS * sizeof(float);
    fused_stats.memory_bytes_written = conv_out_size * sizeof(float);
    fused_stats.compute_metrics();
    stats_array[(*stats_idx)++] = fused_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 3-5: MaxPool, Flatten, FC (same as non-fused)
    // ========================================================================
    int pool_blocks = (pool_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    maxpool2x2_kernel<<<pool_blocks, THREADS_PER_BLOCK>>>(
        model->d_normalized, model->d_pooled,
        batch_size, NUM_CONV_FILTERS, CONV_OUT_H, CONV_OUT_W, POOL_OUT_H, POOL_OUT_W
    );
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaMemcpy(model->d_fc_in, model->d_pooled,
                         pool_out_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    int fc_out_size = batch_size * NUM_CLASSES;
    int fc_blocks = (fc_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fc_forward_kernel<<<fc_blocks, THREADS_PER_BLOCK>>>(
        model->d_fc_in, model->d_fc_weights, model->d_fc_bias, model->d_logits,
        batch_size, FC_INPUT_DIM, NUM_CLASSES
    );
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------
// MODE 2: FUSED V2 (EVONORM-B0)
// ----------------------------------------------------------------------------
// Uses EvoNorm-B0 instead of separate activation + normalization
// Conv → [EvoNorm-B0] → MaxPool → FC
//
// This is a novel normalization-activation layer from neural architecture search
// No explicit activation function!
// ----------------------------------------------------------------------------

void forward_pass_fused_v2(
    ModelParams* model,
    const float* d_input,
    const unsigned char* labels,
    int batch_size,
    KernelStats* stats_array,
    int* stats_idx
) {
    int conv_out_size = batch_size * NUM_CONV_FILTERS * CONV_OUT_H * CONV_OUT_W;
    int pool_out_size = batch_size * NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
    
    CudaTimer timer;
    
    // ========================================================================
    // STEP 1: Convolution
    // ========================================================================
    int conv_blocks = (conv_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    conv2d_simple_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(
        d_input, model->d_conv_weights, model->d_conv_bias, model->d_conv_out,
        batch_size, NUM_INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
        NUM_CONV_FILTERS, KERNEL_SIZE, CONV_PADDING, CONV_STRIDE,
        CONV_OUT_H, CONV_OUT_W
    );
    timer.stop();
    
    KernelStats conv_stats("Conv2D [V2]");
    conv_stats.execution_time_ms = timer.get_elapsed_ms();
    conv_stats.memory_bytes_read = batch_size * IMAGE_SIZE * IMAGE_SIZE * sizeof(float) +
                                    NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    conv_stats.memory_bytes_written = conv_out_size * sizeof(float);
    conv_stats.compute_metrics();
    stats_array[(*stats_idx)++] = conv_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 2: EvoNorm-B0
    // ========================================================================
    int evonorm_total = batch_size * NUM_CONV_FILTERS;
    int evonorm_blocks = (evonorm_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    evonorm_b0_kernel<<<evonorm_blocks, THREADS_PER_BLOCK>>>(
        model->d_conv_out,
        model->d_normalized,
        model->d_evonorm_v1,
        model->d_evonorm_gamma,
        batch_size,
        NUM_CONV_FILTERS,
        CONV_OUT_H,
        CONV_OUT_W,
        LAYERNORM_EPS
    );
    timer.stop();
    
    KernelStats evonorm_stats("EvoNorm-B0 [V2]");
    evonorm_stats.execution_time_ms = timer.get_elapsed_ms();
    evonorm_stats.memory_bytes_read = conv_out_size * sizeof(float) + 
                                       2 * NUM_CONV_FILTERS * sizeof(float);
    evonorm_stats.memory_bytes_written = conv_out_size * sizeof(float);
    evonorm_stats.compute_metrics();
    stats_array[(*stats_idx)++] = evonorm_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 3-5: MaxPool, Flatten, FC
    // ========================================================================
    int pool_blocks = (pool_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    maxpool2x2_kernel<<<pool_blocks, THREADS_PER_BLOCK>>>(
        model->d_normalized, model->d_pooled,
        batch_size, NUM_CONV_FILTERS, CONV_OUT_H, CONV_OUT_W, POOL_OUT_H, POOL_OUT_W
    );
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaMemcpy(model->d_fc_in, model->d_pooled,
                         pool_out_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    int fc_out_size = batch_size * NUM_CLASSES;
    int fc_blocks = (fc_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fc_forward_kernel<<<fc_blocks, THREADS_PER_BLOCK>>>(
        model->d_fc_in, model->d_fc_weights, model->d_fc_bias, model->d_logits,
        batch_size, FC_INPUT_DIM, NUM_CLASSES
    );
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------
// MODE 3: FUSED V3 (SWISH + LAYERNORM)
// ----------------------------------------------------------------------------
// Similar to V1 but uses Swish activation instead of GELU
// Conv → [Swish+LayerNorm] → MaxPool → FC
// ----------------------------------------------------------------------------

void forward_pass_fused_v3(
    ModelParams* model,
    const float* d_input,
    const unsigned char* labels,
    int batch_size,
    KernelStats* stats_array,
    int* stats_idx
) {
    int conv_out_size = batch_size * NUM_CONV_FILTERS * CONV_OUT_H * CONV_OUT_W;
    int pool_out_size = batch_size * NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
    
    CudaTimer timer;
    
    // ========================================================================
    // STEP 1: Convolution
    // ========================================================================
    int conv_blocks = (conv_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    timer.start();
    conv2d_simple_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(
        d_input, model->d_conv_weights, model->d_conv_bias, model->d_conv_out,
        batch_size, NUM_INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
        NUM_CONV_FILTERS, KERNEL_SIZE, CONV_PADDING, CONV_STRIDE,
        CONV_OUT_H, CONV_OUT_W
    );
    timer.stop();
    
    KernelStats conv_stats("Conv2D [V3]");
    conv_stats.execution_time_ms = timer.get_elapsed_ms();
    conv_stats.memory_bytes_read = batch_size * IMAGE_SIZE * IMAGE_SIZE * sizeof(float) +
                                    NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    conv_stats.memory_bytes_written = conv_out_size * sizeof(float);
    conv_stats.compute_metrics();
    stats_array[(*stats_idx)++] = conv_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 2: FUSED SWISH + LayerNorm
    // ========================================================================
    int fused_blocks = batch_size;
    int shared_mem_size = 2 * THREADS_PER_BLOCK * sizeof(float);
    
    timer.start();
    swish_layernorm_fused_kernel<<<fused_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        model->d_conv_out,
        model->d_normalized,
        model->d_ln_gamma,
        model->d_ln_beta,
        batch_size,
        NUM_CONV_FILTERS,
        CONV_OUT_H,
        CONV_OUT_W,
        SWISH_BETA,
        LAYERNORM_EPS
    );
    timer.stop();
    
    KernelStats fused_stats("Swish+LayerNorm FUSED [V3]");
    fused_stats.execution_time_ms = timer.get_elapsed_ms();
    fused_stats.memory_bytes_read = conv_out_size * sizeof(float) + 
                                      2 * NUM_CONV_FILTERS * sizeof(float);
    fused_stats.memory_bytes_written = conv_out_size * sizeof(float);
    fused_stats.compute_metrics();
    stats_array[(*stats_idx)++] = fused_stats;
    
    cudaDeviceSynchronize();
    
    // ========================================================================
    // STEP 3-5: MaxPool, Flatten, FC
    // ========================================================================
    int pool_blocks = (pool_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    maxpool2x2_kernel<<<pool_blocks, THREADS_PER_BLOCK>>>(
        model->d_normalized, model->d_pooled,
        batch_size, NUM_CONV_FILTERS, CONV_OUT_H, CONV_OUT_W, POOL_OUT_H, POOL_OUT_W
    );
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaMemcpy(model->d_fc_in, model->d_pooled,
                         pool_out_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    int fc_out_size = batch_size * NUM_CLASSES;
    int fc_blocks = (fc_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fc_forward_kernel<<<fc_blocks, THREADS_PER_BLOCK>>>(
        model->d_fc_in, model->d_fc_weights, model->d_fc_bias, model->d_logits,
        batch_size, FC_INPUT_DIM, NUM_CLASSES
    );
    cudaDeviceSynchronize();
}



// ============================================================================
// INITIALIZATION AND MEMORY MANAGEMENT
// ============================================================================

ModelParams* allocate_model() {
    ModelParams* model = (ModelParams*)malloc(sizeof(ModelParams));
    
    // Allocate all GPU memory
    CHECK_CUDA(cudaMalloc(&model->d_conv_weights, 
                         NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv_bias, NUM_CONV_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_ln_gamma, NUM_CONV_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_ln_beta, NUM_CONV_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_evonorm_v1, NUM_CONV_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_evonorm_gamma, NUM_CONV_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc_weights, NUM_CLASSES * FC_INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc_bias, NUM_CLASSES * sizeof(float)));
    
    // Allocate intermediate buffers
    int conv_out_size = BATCH_SIZE * NUM_CONV_FILTERS * CONV_OUT_H * CONV_OUT_W;
    int pool_out_size = BATCH_SIZE * NUM_CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
    
    CHECK_CUDA(cudaMalloc(&model->d_conv_out, conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_activated, conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_normalized, conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_pooled, pool_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc_in, BATCH_SIZE * FC_INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_logits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    
    return model;
}

void initialize_model(ModelParams* model) {
    // Initialize weights on CPU then copy to GPU
    int conv_w_size = NUM_CONV_FILTERS * NUM_INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    float* h_conv_w = (float*)malloc(conv_w_size * sizeof(float));
    init_weights(h_conv_w, conv_w_size);
    CHECK_CUDA(cudaMemcpy(model->d_conv_weights, h_conv_w, 
                         conv_w_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv_w);
    
    // Initialize biases to zero
    CHECK_CUDA(cudaMemset(model->d_conv_bias, 0, NUM_CONV_FILTERS * sizeof(float)));
    
    // Initialize LayerNorm parameters
    init_layernorm_params_kernel<<<(NUM_CONV_FILTERS + 255) / 256, 256>>>(
        model->d_ln_gamma, model->d_ln_beta, NUM_CONV_FILTERS
    );
    
    // Initialize EvoNorm parameters
    init_evonorm_params_kernel<<<(NUM_CONV_FILTERS + 255) / 256, 256>>>(
        model->d_evonorm_v1, model->d_evonorm_gamma, NUM_CONV_FILTERS
    );
    
    // Initialize FC layer
    int fc_w_size = NUM_CLASSES * FC_INPUT_DIM;
    float* h_fc_w = (float*)malloc(fc_w_size * sizeof(float));
    init_weights(h_fc_w, fc_w_size);
    CHECK_CUDA(cudaMemcpy(model->d_fc_weights, h_fc_w, 
                         fc_w_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fc_w);
    
    CHECK_CUDA(cudaMemset(model->d_fc_bias, 0, NUM_CLASSES * sizeof(float)));
    
    cudaDeviceSynchronize();
}

void free_model(ModelParams* model) {
    cudaFree(model->d_conv_weights);
    cudaFree(model->d_conv_bias);
    cudaFree(model->d_ln_gamma);
    cudaFree(model->d_ln_beta);
    cudaFree(model->d_evonorm_v1);
    cudaFree(model->d_evonorm_gamma);
    cudaFree(model->d_fc_weights);
    cudaFree(model->d_fc_bias);
    cudaFree(model->d_conv_out);
    cudaFree(model->d_activated);
    cudaFree(model->d_normalized);
    cudaFree(model->d_pooled);
    cudaFree(model->d_fc_in);
    cudaFree(model->d_logits);
    free(model);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    int mode = 5;  // Default: compare all
    if (argc > 1) {
        mode = atoi(argv[1]);
    }
    
    printf("=======================================================\n");
    printf("CNN with Advanced Components & Kernel Fusion\n");
    printf("=======================================================\n");
    printf("Architecture: Conv → [Act+Norm] → MaxPool → FC\n");
    printf("Dataset: MNIST (28x28 grayscale images)\n");
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("Conv filters: %d (3x3)\n", NUM_CONV_FILTERS);
    printf("=======================================================\n\n");
    
    // Print GPU info
    print_device_info();
    
    // Load MNIST data
    printf("\nLoading MNIST dataset...\n");
    int total_images, img_h, img_w;
    unsigned char* train_images_u8 = load_idx_images(
        "/kaggle/input/mnist-dataset/train-images.idx3-ubyte",
        &total_images, &img_h, &img_w
    );
    
    int total_labels;
    unsigned char* train_labels = load_idx_labels(
        "/kaggle/input/mnist-dataset/train-labels.idx1-ubyte",
        &total_labels
    );
    
    if (!train_images_u8 || !train_labels) {
        printf("Failed to load MNIST data\n");
        return 1;
    }
    
    printf("Loaded %d images (%dx%d)\n", total_images, img_h, img_w);
    
    // Convert and normalize
    float* h_input = (float*)malloc(BATCH_SIZE * img_h * img_w * sizeof(float));
    convert_images_u8_to_f32(train_images_u8, h_input, BATCH_SIZE, img_h, img_w);
    
    unsigned char* h_labels = (unsigned char*)malloc(BATCH_SIZE);
    memcpy(h_labels, train_labels, BATCH_SIZE);
    
    // Free original data
    free(train_images_u8);
    free(train_labels);
    
    // Copy input to GPU
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, BATCH_SIZE * img_h * img_w * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, BATCH_SIZE * img_h * img_w * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Allocate and initialize model
    printf("\nInitializing model...\n");
    ModelParams* model = allocate_model();
    initialize_model(model);
    
    // Run requested mode
    KernelStats stats_array[20];
    int stats_idx = 0;
    
    CPUTimer end_to_end_timer;
    
    switch(mode) {
        case 0:
            printf("\n=== MODE 0: NON-FUSED (BASELINE) ===\n");
            end_to_end_timer.start();
            forward_pass_non_fused(model, d_input, h_labels, BATCH_SIZE, stats_array, &stats_idx);
            end_to_end_timer.stop();
            printf("\nEnd-to-end inference time: %.4f ms\n", end_to_end_timer.get_elapsed_ms());
            break;
            
        case 1:
            printf("\n=== MODE 1: FUSED V1 (GELU + LayerNorm) ===\n");
            end_to_end_timer.start();
            forward_pass_fused_v1(model, d_input, h_labels, BATCH_SIZE, stats_array, &stats_idx);
            end_to_end_timer.stop();
            printf("\nEnd-to-end inference time: %.4f ms\n", end_to_end_timer.get_elapsed_ms());
            break;
            
        case 2:
            printf("\n=== MODE 2: FUSED V2 (EvoNorm-B0) ===\n");
            end_to_end_timer.start();
            forward_pass_fused_v2(model, d_input, h_labels, BATCH_SIZE, stats_array, &stats_idx);
            end_to_end_timer.stop();
            printf("\nEnd-to-end inference time: %.4f ms\n", end_to_end_timer.get_elapsed_ms());
            break;
            
        case 3:
            printf("\n=== MODE 3: FUSED V3 (Swish + LayerNorm) ===\n");
            end_to_end_timer.start();
            forward_pass_fused_v3(model, d_input, h_labels, BATCH_SIZE, stats_array, &stats_idx);
            end_to_end_timer.stop();
            printf("\nEnd-to-end inference time: %.4f ms\n", end_to_end_timer.get_elapsed_ms());
            break;
            
        case 5:
        default:
            printf("\n=== MODE 5: COMPARE ALL IMPLEMENTATIONS ===\n");
            
            // Run all modes and collect stats
            printf("\nRunning non-fused baseline...\n");
            int base_idx = 0;
            end_to_end_timer.start();
            forward_pass_non_fused(model, d_input, h_labels, BATCH_SIZE, stats_array, &base_idx);
            end_to_end_timer.stop();
            double baseline_time = end_to_end_timer.get_elapsed_ms();
            
            printf("\nRunning GELU+LayerNorm fusion...\n");
            int v1_idx = base_idx;
            end_to_end_timer.start();
            forward_pass_fused_v1(model, d_input, h_labels, BATCH_SIZE, stats_array, &v1_idx);
            end_to_end_timer.stop();
            double v1_time = end_to_end_timer.get_elapsed_ms();
            
            printf("\nRunning EvoNorm-B0...\n");
            int v2_idx = v1_idx;
            end_to_end_timer.start();
            forward_pass_fused_v2(model, d_input, h_labels, BATCH_SIZE, stats_array, &v2_idx);
            end_to_end_timer.stop();
            double v2_time = end_to_end_timer.get_elapsed_ms();
            
            printf("\nRunning Swish+LayerNorm fusion...\n");
            int v3_idx = v2_idx;
            end_to_end_timer.start();
            forward_pass_fused_v3(model, d_input, h_labels, BATCH_SIZE, stats_array, &v3_idx);
            end_to_end_timer.stop();
            double v3_time = end_to_end_timer.get_elapsed_ms();
            
            // Print comparison
            printf("\n");
            printf("=================================================================\n");
            printf("END-TO-END INFERENCE TIME COMPARISON\n");
            printf("=================================================================\n");
            printf("%-40s %12s %12s\n", "Implementation", "Time (ms)", "Speedup");
            printf("-----------------------------------------------------------------\n");
            printf("%-40s %12.4f %12.2fx\n", "Non-fused (Baseline)", baseline_time, 1.0);
            printf("%-40s %12.4f %12.2fx\n", "GELU+LayerNorm Fusion", v1_time, baseline_time/v1_time);
            printf("%-40s %12.4f %12.2fx\n", "EvoNorm-B0", v2_time, baseline_time/v2_time);
            printf("%-40s %12.4f %12.2fx\n", "Swish+LayerNorm Fusion", v3_time, baseline_time/v3_time);
            printf("=================================================================\n");
            break;
    }
    
    // Print individual kernel stats
    if (stats_idx > 0) {
        printf("\n");
        print_comparison_table(stats_array, stats_idx);
    }
    
    // Cleanup
    free_model(model);
    cudaFree(d_input);
    free(h_input);
    free(h_labels);
    
    printf("\n✓ All done!\n\n");
    return 0;
}
