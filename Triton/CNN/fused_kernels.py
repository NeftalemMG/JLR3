%%writefile fused_kernels.py

import triton_fix

import torch
import triton
import triton.language as tl

# GELU + Layernorm Fused

@triton.jit
def gelu_layernorm_fused_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    num_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    
    # Pass 1: Apply GELU and compute mean
    mean = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        
        # GELU
        x_cubed = x * x * x
        inner = x + coeff * x_cubed
        tanh_arg = sqrt_2_over_pi * inner
        gelu = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))
        
        mean += tl.sum(gelu)
    
    mean = mean / num_features
    
    # Pass 2: Compute variance
    var = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        
        # Recompute GELU
        x_cubed = x * x * x
        inner = x + coeff * x_cubed
        tanh_arg = sqrt_2_over_pi * inner
        gelu = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))
        
        diff = gelu - mean
        var += tl.sum(diff * diff)
    
    var = var / num_features
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Pass 3: Apply LayerNorm
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        
        # Recompute GELU one final time
        x_cubed = x * x * x
        inner = x + coeff * x_cubed
        tanh_arg = sqrt_2_over_pi * inner
        gelu = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))
        
        normalized = (gelu - mean) * rstd
        output = gamma * normalized + beta
        
        tl.store(output_ptr + idx, output, mask=mask)

def gelu_layernorm_fused(input_tensor, gamma, beta, eps=1e-5):
    """GELU + LayerNorm fused into single kernel"""
    batch_size = input_tensor.shape[0]
    num_features = input_tensor[0].numel()
    
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    gelu_layernorm_fused_kernel[grid](
        input_tensor, output, gamma, beta,
        batch_size, num_features, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Swish + Layernorm Fused

@triton.jit
def swish_layernorm_fused_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    num_features,
    swish_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Pass 1: Apply Swish and compute mean
    mean = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        
        # Swish
        sigmoid = 1.0 / (1.0 + tl.math.exp(-swish_beta * x))
        swish = x * sigmoid
        
        mean += tl.sum(swish)
    
    mean = mean / num_features
    
    # Pass 2: Compute variance
    var = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        
        # Recompute Swish
        sigmoid = 1.0 / (1.0 + tl.math.exp(-swish_beta * x))
        swish = x * sigmoid
        
        diff = swish - mean
        var += tl.sum(diff * diff)
    
    var = var / num_features
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Pass 3: Apply LayerNorm
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        
        # Recompute Swish one final time
        sigmoid = 1.0 / (1.0 + tl.math.exp(-swish_beta * x))
        swish = x * sigmoid
        
        normalized = (swish - mean) * rstd
        output = gamma * normalized + beta
        
        tl.store(output_ptr + idx, output, mask=mask)

def swish_layernorm_fused(input_tensor, gamma, beta, swish_beta=1.0, eps=1e-5):
    """Swish + LayerNorm fused into single kernel"""
    batch_size = input_tensor.shape[0]
    num_features = input_tensor[0].numel()
    
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    swish_layernorm_fused_kernel[grid](
        input_tensor, output, gamma, beta,
        batch_size, num_features, swish_beta, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output