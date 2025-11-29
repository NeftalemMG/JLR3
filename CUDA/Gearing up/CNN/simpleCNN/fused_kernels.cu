// Fused Kernels: Optimized Implementations
// This file contains 4 different kernel fusion techniques that combine
// multiple operations into single kernels for improved performance.
//
// Kernel Fusion Techniques Implemented:
// 1. Fusion V1: GELU + LayerNorm (Simple Fusion)
// 2. Fusion V2: EvoNorm-B0 (From NeurIPS 2020 paper)
// 3. Fusion V3: Swish + LayerNorm
// 4. Fusion V4: Convolution + GELU + LayerNorm (Complex Fusion)
//
// Why Kernel Fusion?
// Benefits of kernel fusion:
// 1. Reduced Memory Bandwidth: Instead of writing intermediate results to
//    global memory and reading them back, we keep data in registers/shared
//    memory within a single kernel.
//
// 2. Improved Cache Utilization: Data stays hot in cache across operations
//    instead of being evicted between kernel launches.
//
// 3. Reduced Kernel Launch Overhead: Each kernel launch has overhead
//    (scheduling, synchronization). Fusing reduces total launches.
//
// 4. Better Instruction Level Parallelism: Modern GPUs can execute multiple
//    instructions simultaneously. Fusion enables better pipelining.
//
// Example:
//   Non-fused: Write to memory → Read from memory → Write to memory
//   Fused: Compute → Compute → Write to memory (1 write instead of 2)
//
// Trade-offs:
// - Increased register usage (may reduce occupancy)
// - More complex code (harder to debug)
// - May not always be beneficial (depends on operation patterns)


#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


// Fusion V1: GELU + LAYER NORMALIZATION
// This is a simple but effective fusion that combines GELU activation with
// LayerNorm. This pattern is extremely common in transformers:
//   Feed-Forward Network: Linear → GELU → Linear → LayerNorm
//
// Non-fused version requires:
//   1. GELU kernel: Read input, compute GELU, write to temp buffer
//   2. LayerNorm kernel: Read temp buffer, compute norm, write to output
//   Total: 2 kernel launches, 2 reads + 2 writes to global memory
//
// Fused version:
//   1. Combined kernel: Read input, compute GELU, compute norm, write output
//   Total: 1 kernel launch, 1 read + 1 write to global memory
//
// Expected speedup: 1.5-2x (depends on problem size and memory bandwidth)
//
// Algorithm:
//   1. Apply GELU activation element-wise (keep in registers)
//   2. Compute mean and variance of GELU outputs (shared memory reduction)
//   3. Normalize using computed statistics
//   4. Apply learnable scale/shift parameters
//
// This fusion is particularly effective because:
// - GELU is element-wise (no data dependencies)
// - Both operations read same input region
// - Intermediate GELU result stays in registers

__global__ void gelu_layernorm_fused_kernel(
    const float* input,      // Input tensor [N, C, H, W]
    float* output,           // Output tensor [N, C, H, W]
    const float* gamma,      // Scale parameter [C]
    const float* beta,       // Shift parameter [C]
    int N,                   // Batch size
    int C,                   // Number of channels
    int H,                   // Height
    int W,                   // Width
    float eps                // Epsilon for LayerNorm
) {
    // Each block processes one sample
    int n = blockIdx.x;
    if (n >= N) return;
    
    int num_features = C * H * W;
    
    // Shared memory for reduction operations
    extern __shared__ float shared_data[];
    float* shared_sum = &shared_data[0];
    float* shared_sq_sum = &shared_data[blockDim.x];
    
    int tid = threadIdx.x;
    
    // GELU constants
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    // Step 1: Apply GELU and compute mean (FUSED!)
    // Key optimization: We compute GELU and accumulate for mean in one pass
    // The GELU output is stored temporarily in a local variable, not memory
    
    float local_sum = 0.0f;
    
    // Process elements in chunks
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        float x = input[idx];
        
        // Compute GELU(x) - keeping result in register
        float x_cubed = x * x * x;
        float inner = x + coeff * x_cubed;
        float arg = sqrt_2_over_pi * inner;
        float gelu_output = 0.5f * x * (1.0f + tanhf(arg));
        
        // Accumulate for mean calculation
        local_sum += gelu_output;
        
        // Store GELU output temporarily in output buffer
        // (we'll overwrite this with final normalized values)
        output[idx] = gelu_output;
    }
    
    // Reduce to compute mean
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (tid == 0) {
        mean = shared_sum[0] / num_features;
        shared_sum[0] = mean;
    }
    __syncthreads();
    mean = shared_sum[0];
    
    // Step 2: Compute variance

    float local_sq_sum = 0.0f;
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        float gelu_val = output[idx]; // Read GELU output we computed earlier
        float diff = gelu_val - mean;
        local_sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = local_sq_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }
    
    float variance = 0.0f;
    if (tid == 0) {
        variance = shared_sq_sum[0] / num_features;
        shared_sq_sum[0] = variance;
    }
    __syncthreads();
    variance = shared_sq_sum[0];

    // STEP 3: Normalize with scale and shift
    float inv_std = rsqrtf(variance + eps);
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        int c = (i / (H * W)) % C;
        
        float gelu_val = output[idx];
        float normalized = (gelu_val - mean) * inv_std;
        
        // Apply affine transformation
        output[idx] = gamma[c] * normalized + beta[c];
    }
    
    // Performance notes:
    // - GELU computation happens once (not stored to global memory)
    // - Two passes through data (unavoidable for mean/variance)
    // - Compared to non-fused: saves 1 global memory write + 1 global read
}

// Fusion V2: EVONORM-B0 (From "Evolving Normalization-Activation Layers")
// EvoNorm-B0 is a novel normalization-activation layer discovered through
// neural architecture search. It outperforms BatchNorm-ReLU in many cases.
//
// Paper: "Evolving Normalization-Activation Layers" (NeurIPS 2020)
// Authors: Liu et al., Google Research
//
// Formula: EvoNorm-B0(x) = x / max(v1 * σ_instance(x) + σ_batch(x), ε) + γ
//
// where:
//   σ_batch(x) = √(variance across batch, height, width for each channel)
//   σ_instance(x) = √(variance across height, width for each sample+channel)
//   v1 = learnable parameter (per channel)
//   γ = learnable parameter (per channel)
//   ε = small constant (1e-5)
//
// Key innovations of EvoNorm-B0:
// 1. NO explicit activation function (nonlinearity comes from max operation)
// 2. NO mean subtraction (only uses variance for normalization)
// 3. Combines batch and instance statistics in a novel way
// 4. Learns to weight batch vs instance normalization (via v1 parameter)
//
// Why is this interesting?
// - Breaks traditional design patterns (no mean centering!)
// - Self-discovered by automated search (not human intuition)
// - Works well across different architectures (ResNets, MobileNets, etc.)
// - Transfers to other tasks (segmentation, image synthesis)
//
// This implementation is a simplified version for demonstration.
// Full implementation would require batch statistics computation.

__global__ void evonorm_b0_kernel(
    const float* input,       // Input tensor [N, C, H, W]
    float* output,            // Output tensor [N, C, H, W]
    const float* v1,          // Learnable weight for instance var [C]
    const float* gamma,       // Learnable shift parameter [C]
    int N,                    // Batch size
    int C,                    // Number of channels
    int H,                    // Height
    int W,                    // Width
    float eps                 // Small constant for stability
) {
    // Each block processes one channel for one sample
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    
    if (global_id >= total) return;
    
    // Decode indices
    int c = global_id % C;
    int n = global_id / C;
    
    int spatial_size = H * W;
    
    // Step 1: Compute instance variance (within this sample+channel)
    // Instance variance: var = (1/HW) * Σ(x - mean)²
    // For EvoNorm-B0, we can use raw second moment for simplicity
    
    float sum_sq = 0.0f;
    for (int hw = 0; hw < spatial_size; hw++) {
        int idx = ((n * C + c) * H * W) + hw;
        float val = input[idx];
        sum_sq += val * val;
    }
    float instance_var = sum_sq / spatial_size;
    float instance_std = sqrtf(instance_var + eps);
    
    // Step 2: Compute normalization denominator
    // In full EvoNorm-B0: denominator = max(v1 * σ_instance + σ_batch, ε)
    // Simplified version: denominator = max(v1[c] * σ_instance, ε)
    //
    // The max operation provides the nonlinearity (no explicit activation!)
    
    float denominator = fmaxf(v1[c] * instance_std, eps);
    
    // Step 3: Apply normalization and shift
    // output = (input / denominator) + γ
    // 
    // This is different from LayerNorm:
    // - LayerNorm: (x - μ) / σ  (centers then scales)
    // - EvoNorm: x / (v1*σ)  (only scales, no centering!)
    
    for (int hw = 0; hw < spatial_size; hw++) {
        int idx = ((n * C + c) * H * W) + hw;
        float val = input[idx];
        
        // Normalize (divide by denominator)
        float normalized = val / denominator;
        
        // Add learnable shift (note: NOT multiply like LayerNorm!)
        output[idx] = normalized + gamma[c];
    }
    
    // Insights from EvoNorm-B0:
    // 1. Mean centering isn't always necessary
    // 2. The max() operation can provide sufficient nonlinearity
    // 3. Combining batch and instance statistics can be beneficial
    // 4. Automated search can find surprising patterns
    //
    // Performance characteristics:
    // - Fewer operations than BatchNorm-ReLU (no activation function)
    // - No batch statistics needed in inference (faster)
    // - Still maintains gradient flow (important for training)
}

// Fusion V3: SWISH + LAYER NORMALIZATION
// This fusion combines Swish (SiLU) activation with LayerNorm.
// Pattern commonly seen in: EfficientNet, Mobile architectures
//
// Swish recap: f(x) = x * sigmoid(βx)
// LayerNorm recap: y = γ * (x - μ) / σ + β
//
// Combined: We apply Swish first, then normalize the activated features
//
// Benefits of this fusion:
// 1. Swish is smooth (good for optimization)
// 2. LayerNorm stabilizes features (good for deep networks)
// 3. Together they provide both nonlinearity and normalization
//
// This is similar to GELU+LayerNorm fusion but with different activation.
// Swish tends to work better for mobile/efficient architectures.

__global__ void swish_layernorm_fused_kernel(
    const float* input,      // Input tensor [N, C, H, W]
    float* output,           // Output tensor [N, C, H, W]
    const float* gamma,      // Scale parameter [C]
    const float* beta,       // Shift parameter [C]
    int N,                   // Batch size
    int C,                   // Number of channels
    int H,                   // Height
    int W,                   // Width
    float swish_beta,        // Beta for Swish activation
    float eps                // Epsilon for LayerNorm
) {
    int n = blockIdx.x;
    if (n >= N) return;
    
    int num_features = C * H * W;
    
    extern __shared__ float shared_data[];
    float* shared_sum = &shared_data[0];
    float* shared_sq_sum = &shared_data[blockDim.x];
    
    int tid = threadIdx.x;
    
    // Step 1: Apply Swish and compute mean (FUSED!)
    float local_sum = 0.0f;
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        float x = input[idx];
        
        // Compute Swish(x) = x * sigmoid(β*x)
        float sigmoid = 1.0f / (1.0f + expf(-swish_beta * x));
        float swish_output = x * sigmoid;
        
        // Accumulate for mean
        local_sum += swish_output;
        
        // Store Swish output temporarily
        output[idx] = swish_output;
    }
    
    // Reduce to compute mean
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (tid == 0) {
        mean = shared_sum[0] / num_features;
        shared_sum[0] = mean;
    }
    __syncthreads();
    mean = shared_sum[0];
    
    // Step 2: Compute variance
    float local_sq_sum = 0.0f;
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        float swish_val = output[idx];
        float diff = swish_val - mean;
        local_sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = local_sq_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }
    
    float variance = 0.0f;
    if (tid == 0) {
        variance = shared_sq_sum[0] / num_features;
        shared_sq_sum[0] = variance;
    }
    __syncthreads();
    variance = shared_sq_sum[0];
    
    // Step 3: Normalize with scale and shift
    float inv_std = rsqrtf(variance + eps);
    
    for (int i = tid; i < num_features; i += blockDim.x) {
        int idx = n * num_features + i;
        int c = (i / (H * W)) % C;
        
        float swish_val = output[idx];
        float normalized = (swish_val - mean) * inv_std;
        
        output[idx] = gamma[c] * normalized + beta[c];
    }
    
    // Comparison with ReLU+LayerNorm:
    // - Swish provides smoother gradients
    // - Swish can output small negative values (richer representations)
    // - Swish adds minimal overhead vs ReLU (one sigmoid computation)
}

// Fusion V4: Convolution + GELU + Layer Normalization (Complex Fusion)
// This is the most aggressive fusion: combining three operations into one!
//
// Typical sequence in modern CNNs:
//   Conv2D → Activation → Normalization
//
// Without fusion:
//   1. Conv kernel: Computes convolution, writes to temp1
//   2. Activation kernel: Reads temp1, applies GELU, writes to temp2
//   3. LayerNorm kernel: Reads temp2, normalizes, writes to output
//   Total: 3 kernels, 3 writes + 2 reads
//
// With fusion:
//   1. Mega-kernel: Computes conv, applies GELU, normalizes, writes output
//   Total: 1 kernel, 1 write
//
// Expected speedup: 2-3x (significant memory savings)
//
// Challenges:
// - High register pressure (may reduce occupancy)
// - Complex control flow
// - Limited by convolution compute requirements
//
// When to use:
// - Small kernels (3x3 or smaller)
// - When memory bandwidth is the bottleneck
// - Modern GPUs with large register files
//
// Implementation strategy:
// - Each thread computes one output element
// - Load input patch, apply conv weights, apply GELU (keep in registers)
// - Use shared memory for normalization statistics

__global__ void conv_gelu_layernorm_fused_kernel(
    const float* input,      // Input tensor [N, C_in, H, W]
    const float* weight,     // Conv weights [C_out, C_in, K, K]
    const float* bias,       // Conv bias [C_out]
    float* output,           // Output tensor [N, C_out, H_out, W_out]
    const float* gamma,      // LayerNorm scale [C_out]
    const float* beta,       // LayerNorm shift [C_out]
    int N,                   // Batch size
    int C_in,                // Input channels
    int H, int W,            // Input height and width
    int C_out,               // Output channels
    int K,                   // Kernel size
    int pad, int stride,     // Padding and stride
    int H_out, int W_out,    // Output height and width
    float eps                // Epsilon for LayerNorm
) {
    // This kernel is organized differently than previous ones:
    // - Each BLOCK processes one sample in the batch
    // - Threads within block handle spatial locations and channels
    
    int n = blockIdx.x;
    if (n >= N) return;
    
    extern __shared__ float shared_mem[];
    float* conv_output_shared = shared_mem; // Store conv outputs temporarily
    float* shared_stats = &shared_mem[C_out * H_out * W_out]; // For mean/var
    
    int tid = threadIdx.x;
    int total_output_elements = C_out * H_out * W_out;
    
    // GELU constants
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Step 1: Convolution (each thread computes multiple outputs)
    for (int idx = tid; idx < total_output_elements; idx += blockDim.x) {
        // Decode output position
        int x = idx % W_out;
        int tmp = idx / W_out;
        int y = tmp % H_out;
        int m = tmp / H_out; // output channel
        
        // Compute convolution for this output position
        float acc = 0.0f;
        for (int c = 0; c < C_in; c++) {
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    int in_y = y * stride + ky - pad;
                    int in_x = x * stride + kx - pad;
                    
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int in_idx = ((n * C_in + c) * H + in_y) * W + in_x;
                        int w_idx = ((m * C_in + c) * K + ky) * K + kx;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        
        // Add bias
        if (bias) {
            acc += bias[m];
        }
        
        // Step 2: GELU Activation (applied immediately, kept in registers)
        float x_val = acc;
        float x_cubed = x_val * x_val * x_val;
        float inner = x_val + coeff * x_cubed;
        float arg = sqrt_2_over_pi * inner;
        float gelu_output = 0.5f * x_val * (1.0f + tanhf(arg));
        
        // Store in shared memory for normalization
        conv_output_shared[idx] = gelu_output;
    }
    __syncthreads();
    
    // Step 3: Layer Normalization
    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < total_output_elements; i += blockDim.x) {
        local_sum += conv_output_shared[i];
    }
    
    // Simplified reduction (can be optimized further)
    atomicAdd(&shared_stats[0], local_sum);
    __syncthreads();
    
    float mean = shared_stats[0] / total_output_elements;
    __syncthreads();
    
    // Compute variance
    float local_sq_sum = 0.0f;
    for (int i = tid; i < total_output_elements; i += blockDim.x) {
        float diff = conv_output_shared[i] - mean;
        local_sq_sum += diff * diff;
    }
    
    atomicAdd(&shared_stats[1], local_sq_sum);
    __syncthreads();
    
    float variance = shared_stats[1] / total_output_elements;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();
    
    // STEP 4: Apply Normalization and Write Output
    for (int idx = tid; idx < total_output_elements; idx += blockDim.x) {
        // Get channel for this element
        int c_out = (idx / (H_out * W_out)) % C_out;
        
        float val = conv_output_shared[idx];
        float normalized = (val - mean) * inv_std;
        float final_output = gamma[c_out] * normalized + beta[c_out];
        
        // Write to global memory (only once!)
        int out_idx = n * total_output_elements + idx;
        output[out_idx] = final_output;
    }
    
    // Performance analysis:
    // NON-FUSED: 3 global writes + 2 global reads = 5 memory operations
    // FUSED: 1 global write + 1 global read = 2 memory operations
    // Memory bandwidth savings: 60%
    //
    // However, trade-offs:
    // - Higher register usage (stores conv weights, intermediate values)
    // - More complex (harder to maintain)
    // - May reduce occupancy if registers are exhausted
    //
    // Best used when: Memory bandwidth > compute bound
}

// HELPER: Initialize EvoNorm parameters
__global__ void init_evonorm_params_kernel(
    float* v1,          // Weight for instance variance [C]
    float* gamma,       // Shift parameter [C]
    int C               // Number of channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    
    v1[c] = 1.0f;      // Initialize to 1
    gamma[c] = 0.0f;   // Initialize to 0
}
