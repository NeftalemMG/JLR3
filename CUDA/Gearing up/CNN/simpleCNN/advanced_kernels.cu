// Non-fused implementations
// This file contains implementations of modern neural network components:
// 1. Layer Normalization (LayerNorm)
// 2. GELU (Gaussian Error Linear Unit) activation
// 3. Swish (SiLU) activation
// 4. Custom loss functions (Focal Loss, Label Smoothing Cross Entropy)
//
// All implementations here are NON-FUSED, meaning each operation runs as
// a separate kernel. This serves as our baseline for performance comparison.

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Layer normalization (Layer Norm)
// LayerNorm normalizes features across the channel dimension for each sample.
// Unlike BatchNorm which normalizes across the batch dimension, LayerNorm
// normalizes across features, making it ideal for:
// - Transformers and attention mechanisms
// - Sequence models where batch statistics are unstable
// - Small batch sizes where BatchNorm struggles
//
// Formula: y = γ * (x - μ) / √(σ² + ε) + β
// where:
//   μ = mean of features for each sample
//   σ² = variance of features for each sample
//   γ = learnable scale parameter (initialized to 1)
//   β = learnable shift parameter (initialized to 0)
//   ε = small constant for numerical stability (typically 1e-5)
//
// Input shape: [N, C, H, W] where:
//   N = batch size
//   C = number of channels/features
//   H = height
//   W = width
//
// LayerNorm normalizes across C * H * W for each sample in the batch

__global__ void layernorm_forward_kernel(
    const float* input,      // Input tensor [N, C, H, W]
    float* output,           // Output tensor [N, C, H, W]
    const float* gamma,      // Scale parameter [C] - learnable
    const float* beta,       // Shift parameter [C] - learnable
    int N,                   // Batch size
    int C,                   // Number of channels
    int H,                   // Height
    int W,                   // Width
    float eps                // Epsilon for numerical stability
) {
    // Each block processes one sample in the batch
    int n = blockIdx.x;
    if (n >= N) return;
    
    // Number of elements to normalize per sample (all features)
    int num_features = C * H * W;
    
    // Shared memory for efficient reduction operations
    // We use shared memory to accumulate partial sums across threads
    extern __shared__ float shared_data[];
    float* shared_sum = &shared_data[0];           // For mean calculation
    float* shared_sq_sum = &shared_data[blockDim.x]; // For variance calculation
    
    int tid = threadIdx.x;
    
    // ========================================================================
    // STEP 1: Calculate Mean (μ)
    // ========================================================================
    // Each thread accumulates a portion of the sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        local_sum += input[idx];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction: sum up all partial sums
    // This is a parallel reduction using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 now has the total sum; compute mean
    float mean = 0.0f;
    if (tid == 0) {
        mean = shared_sum[0] / num_features;
        shared_sum[0] = mean; // Store mean for other threads
    }
    __syncthreads();
    mean = shared_sum[0]; // All threads read the mean
    
    // ========================================================================
    // STEP 2: Calculate Variance (σ²)
    // ========================================================================
    // Variance = E[(x - μ)²]
    float local_sq_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        float diff = input[idx] - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    __syncthreads();
    
    // Reduction: sum up all squared differences
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes variance
    float variance = 0.0f;
    if (tid == 0) {
        variance = shared_sq_sum[0] / num_features;
        shared_sq_sum[0] = variance; // Store variance
    }
    __syncthreads();
    variance = shared_sq_sum[0]; // All threads read variance
    
    // ========================================================================
    // STEP 3: Normalize and Apply Affine Transformation
    // ========================================================================
    // y = γ * (x - μ) / √(σ² + ε) + β
    float inv_std = rsqrtf(variance + eps); // 1/√(σ² + ε)
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        
        // Get which channel this element belongs to
        int c = (i / (H * W)) % C;
        
        // Normalize: (x - μ) / √(σ² + ε)
        float normalized = (input[idx] - mean) * inv_std;
        
        // Apply scale and shift: γ * normalized + β
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// GELU Activation (Gaussian Error Linear Unit)
// GELU is a smooth, non-monotonic activation function that has become the
// standard in transformers and modern language models (BERT, GPT, etc.).
//
// Why GELU instead of ReLU?
// - Smooth gradients: Unlike ReLU's hard cutoff at 0, GELU provides smooth
//   gradients everywhere, which helps optimization
// - Non-monotonic: Can output negative values for negative inputs, providing
//   richer representations
// - Probabilistic interpretation: GELU(x) ≈ x * P(X ≤ x) where X ~ N(0,1)
//   This means it weights inputs by their CDF
//
// Mathematical formulation:
// GELU(x) = x * Φ(x)
// where Φ(x) is the cumulative distribution function of N(0,1)
//
// Approximation (used here for efficiency):
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// This approximation is:
// - Fast to compute (avoids erf function)
// - Accurate (< 0.1% error compared to exact GELU)
// - Widely used in practice (PyTorch, TensorFlow)
//
// Input/Output: Both are [N, C, H, W] tensors (element-wise operation)

__global__ void gelu_activation_kernel(
    const float* input,    // Input tensor [N, C, H, W]
    float* output,         // Output tensor [N, C, H, W]
    int total_elements     // Total number of elements (N * C * H * W)
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    float x = input[idx];
    
    // GELU approximation constants
    const float sqrt_2_over_pi = 0.7978845608f; // √(2/π)
    const float coeff = 0.044715f;
    
    // Compute: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    // 
    // Breaking it down:
    // 1. Compute x³
    float x_cubed = x * x * x;
    
    // 2. Compute inner term: x + 0.044715 * x³
    float inner = x + coeff * x_cubed;
    
    // 3. Multiply by √(2/π)
    float arg = sqrt_2_over_pi * inner;
    
    // 4. Apply tanh
    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    // Using tanhf for single precision
    float tanh_arg = tanhf(arg);
    
    // 5. Final computation: 0.5 * x * (1 + tanh_arg)
    output[idx] = 0.5f * x * (1.0f + tanh_arg);
    
    // Visualization of GELU behavior:
    // x < -1: output ≈ 0 (suppresses large negative values)
    // x ≈ 0: output ≈ 0 (but with smooth transition)
    // x > 1: output ≈ x (passes through large positive values)
    // This creates a smooth, probabilistic gating mechanism
}


// Swish Activation (Also known as SiLU - Sigmoid Linear Unit)
// Swish is a self-gated activation function discovered through neural
// architecture search. It's become popular in EfficientNet and modern CNNs.
//
// Formula: Swish(x) = x * σ(βx)
// where:
//   σ(x) = 1 / (1 + e^(-x)) is the sigmoid function
//   β is a learnable or fixed parameter (typically β=1)
//
// Why Swish?
// - Smooth: Unlike ReLU, it's smooth everywhere (helpful for optimization)
// - Non-monotonic: For small β, it has a "bump" for negative x values
// - Self-gating: The output is modulated by a smooth gate σ(βx)
// - Unbounded above: Like ReLU, it can output arbitrarily large values
// - Bounded below: Unlike ReLU, it approaches 0 smoothly (not hard cutoff)
//
// Properties:
// - When β=0: Swish(x) = x/2 (scaled identity)
// - When β→∞: Swish(x) → ReLU(x) (approaches ReLU)
// - When β=1: Swish(x) = x * sigmoid(x) (most common, called SiLU)
//
// Advantages over ReLU:
// 1. Smooth gradients (no dead neurons)
// 2. Can output small negative values (richer representations)
// 3. Self-gating provides adaptive behavior
//
// Input/Output: Both are [N, C, H, W] tensors (element-wise operation)

__global__ void swish_activation_kernel(
    const float* input,    // Input tensor [N, C, H, W]
    float* output,         // Output tensor [N, C, H, W]
    int total_elements,    // Total number of elements (N * C * H * W)
    float beta             // Beta parameter (typically 1.0)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    float x = input[idx];
    
    // Compute sigmoid(β * x) = 1 / (1 + e^(-βx))
    // We use the identity: sigmoid(x) = 1 / (1 + exp(-x))
    //
    // For numerical stability, we can rewrite this as:
    // If x ≥ 0: sigmoid(x) = 1 / (1 + exp(-x))
    // If x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
    //
    // However, for simplicity and since beta is typically 1,
    // we'll use the standard formulation with expf
    
    float sigmoid = 1.0f / (1.0f + expf(-beta * x));
    
    // Swish(x) = x * sigmoid(βx)
    output[idx] = x * sigmoid;
    
    // Behavior visualization:
    // x << 0: sigmoid ≈ 0, so output ≈ 0 (similar to ReLU)
    // x ≈ 0: sigmoid ≈ 0.5, so output ≈ 0.5x (smooth transition)
    // x >> 0: sigmoid ≈ 1, so output ≈ x (linear passthrough)
    //
    // The key difference from ReLU is the smooth transition around 0,
    // and for negative x, Swish can output small negative values,
    // providing a slight regularization effect
}

// Focal Loss: 
// Focal Loss is designed to address class imbalance in classification tasks.
// It was introduced in the paper "Focal Loss for Dense Object Detection"
// for the RetinaNet architecture.
//
// Standard Cross Entropy: CE(p, y) = -log(p_y)
// where p_y is the predicted probability for the true class y
//
// Focal Loss: FL(p, y) = -α_y * (1 - p_y)^γ * log(p_y)
// where:
//   α_y = balancing factor for class y (addresses class imbalance)
//   γ = focusing parameter (typically 2)
//   p_y = predicted probability for the true class
//
// Key insight: The term (1 - p_y)^γ is the "modulating factor"
// - When an example is misclassified (p_y is small), (1-p_y) ≈ 1,
//   so the loss is unaffected (focuses on hard examples)
// - When an example is well-classified (p_y is large), (1-p_y) ≈ 0,
//   so the loss is down-weighted (easy examples contribute less)
//
// This forces the model to focus on hard, misclassified examples.
//
// Parameters:
//   γ=0: Focal Loss = Cross Entropy (no focusing)
//   γ=2: Typical value (good balance)
//   γ=5: Extreme focusing on hard examples
//
// Example:
//   If p_y = 0.9 (easy example), (1-0.9)^2 = 0.01, loss is scaled by 0.01
//   If p_y = 0.3 (hard example), (1-0.3)^2 = 0.49, loss is scaled by 0.49
//
// This kernel computes Focal Loss given predictions and labels

__global__ void focal_loss_kernel(
    const float* predictions,  // Predicted probabilities [N, num_classes]
    const int* labels,         // Ground truth labels [N]
    float* losses,             // Output loss for each sample [N]
    int N,                     // Batch size
    int num_classes,           // Number of classes
    float alpha,               // Balancing factor (typically 0.25)
    float gamma                // Focusing parameter (typically 2.0)
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N) return;
    
    // Get the true label for this sample
    int true_label = labels[n];
    
    // Get predicted probability for the true class
    float p_t = predictions[n * num_classes + true_label];
    
    // Clamp probability to avoid log(0)
    p_t = fmaxf(p_t, 1e-7f);
    p_t = fminf(p_t, 1.0f - 1e-7f);
    
    // Compute focal loss: -α * (1 - p_t)^γ * log(p_t)
    float focal_weight = powf(1.0f - p_t, gamma);
    losses[n] = -alpha * focal_weight * logf(p_t);
    
    // Intuition: 
    // If the model is confident and correct (p_t ≈ 1):
    //   focal_weight ≈ 0, so loss ≈ 0 (easy example, ignore it)
    // If the model is uncertain or wrong (p_t ≈ 0):
    //   focal_weight ≈ 1, so loss ≈ -log(p_t) (hard example, focus on it)
}

// ============================================================================
// LABEL SMOOTHING CROSS ENTROPY
// ============================================================================
// Label smoothing is a regularization technique that prevents the model from
// becoming overconfident. Instead of using hard labels (0 or 1), we smooth
// them to encourage the model to be less certain.
//
// Standard Cross Entropy uses one-hot labels:
//   True class: y_true = 1
//   Other classes: y_true = 0
//
// Label Smoothing modifies these labels:
//   True class: y_smooth = 1 - ε + ε/K
//   Other classes: y_smooth = ε/K
//
// where:
//   ε = smoothing parameter (typically 0.1)
//   K = number of classes
//
// Example with K=10 classes and ε=0.1:
//   True class: 1 - 0.1 + 0.1/10 = 0.91
//   Other classes: 0.1/10 = 0.01
//
// Loss formula:
//   L = -Σ y_smooth * log(p)
//   L = -(1-ε+ε/K) * log(p_true) - (K-1) * (ε/K) * log(p_other_avg)
//
// Benefits:
// 1. Prevents overconfidence (no predicted probability becomes exactly 1)
// 2. Improves generalization (model doesn't overfit to training labels)
// 3. Better calibration (predicted probabilities match true frequencies)
// 4. Acts as regularization (similar to adding noise to labels)
//
// This kernel computes label-smoothed cross entropy
// ============================================================================

__global__ void label_smoothing_cross_entropy_kernel(
    const float* predictions,  // Predicted probabilities [N, num_classes]
    const int* labels,         // Ground truth labels [N]
    float* losses,             // Output loss for each sample [N]
    int N,                     // Batch size
    int num_classes,           // Number of classes (K)
    float smoothing            // Smoothing parameter ε (typically 0.1)
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N) return;
    
    int true_label = labels[n];
    
    // Smooth label values
    float smooth_positive = 1.0f - smoothing + smoothing / num_classes; // For true class
    float smooth_negative = smoothing / num_classes;                    // For other classes
    
    float loss = 0.0f;
    
    // Compute loss as -Σ y_smooth * log(p)
    for (int c = 0; c < num_classes; c++) {
        float p = predictions[n * num_classes + c];
        
        // Clamp to avoid log(0)
        p = fmaxf(p, 1e-7f);
        
        // Use smooth label
        float y_smooth = (c == true_label) ? smooth_positive : smooth_negative;
        
        // Accumulate: -y_smooth * log(p)
        loss += -y_smooth * logf(p);
    }
    
    losses[n] = loss;
    
    // Interpretation:
    // Without smoothing (ε=0): Loss = -log(p_true_class) (standard CE)
    // With smoothing (ε=0.1): Loss penalizes low probabilities on ALL classes,
    //                         not just the true class, forcing the model to
    //                         maintain reasonable probabilities everywhere
}

// ============================================================================
// HELPER: Initialize gamma and beta for LayerNorm
// ============================================================================
// LayerNorm requires learnable parameters gamma (scale) and beta (shift).
// Standard initialization:
//   gamma = 1.0 (no scaling initially)
//   beta = 0.0 (no shift initially)
//
// During training, these parameters will be learned to optimally scale and
// shift the normalized features for each channel.
// ============================================================================

__global__ void init_layernorm_params_kernel(
    float* gamma,       // Scale parameters [C]
    float* beta,        // Shift parameters [C]
    int C               // Number of channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= C) return;
    
    gamma[c] = 1.0f;  // Initialize to 1 (identity scaling)
    beta[c] = 0.0f;   // Initialize to 0 (no shift)
}

// ============================================================================
// UTILITY: Softmax (for comparison with custom losses)
// ============================================================================
// Standard softmax implementation for converting logits to probabilities.
// This is used as preprocessing for both Focal Loss and Label Smoothing CE.
//
// Softmax formula: p_i = exp(x_i) / Σ exp(x_j)
//
// We use the numerically stable version:
// p_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
//
// Subtracting max(x) prevents overflow from large exponentials
// ============================================================================

__global__ void softmax_kernel(
    const float* input,     // Input logits [N, num_classes]
    float* output,          // Output probabilities [N, num_classes]
    int N,                  // Batch size
    int num_classes         // Number of classes
) {
    int n = blockIdx.x;
    if (n >= N) return;
    
    extern __shared__ float shared_max_exp[];
    float* shared_max = &shared_max_exp[0];
    float* shared_sum = &shared_max_exp[blockDim.x];
    
    int tid = threadIdx.x;
    
    // Find maximum value for numerical stability
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float val = input[n * num_classes + c];
        local_max = fmaxf(local_max, val);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float val = expf(input[n * num_classes + c] - max_val);
        output[n * num_classes + c] = val;
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int c = tid; c < num_classes; c += blockDim.x) {
        output[n * num_classes + c] /= sum;
    }
}
