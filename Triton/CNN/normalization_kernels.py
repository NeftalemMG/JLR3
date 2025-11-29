%%writefile normalization_kernels.py

import triton_fix

import torch
import triton
import triton.language as tl

# Layernorm

@triton.jit
def layernorm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    num_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Compute mean
    mean = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        mean += tl.sum(vals)
    
    mean = mean / num_features
    
    # Compute variance
    var = 0.0
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        diff = vals - mean
        var += tl.sum(diff * diff)
    
    var = var / num_features
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transform
    for i in range(0, num_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        idx = pid * num_features + offsets
        
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        
        normalized = (vals - mean) * rstd
        output = gamma * normalized + beta
        
        tl.store(output_ptr + idx, output, mask=mask)

def layernorm(input_tensor, gamma, beta, eps=1e-5):
    """Layer Normalization"""
    batch_size = input_tensor.shape[0]
    num_features = input_tensor[0].numel()
    
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    layernorm_kernel[grid](
        input_tensor, output, gamma, beta,
        batch_size, num_features, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# EVONORM-B0

@triton.jit
def evonorm_b0_kernel(
    input_ptr,
    output_ptr,
    v1_ptr,
    gamma_ptr,
    batch_size,
    num_channels,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) instance
    pid = tl.program_id(0)
    
    total_instances = batch_size * num_channels
    if pid >= total_instances:
        return
    
    batch_idx = pid // num_channels
    channel_idx = pid % num_channels
    
    # Compute instance standard deviation
    sum_squares = 0.0
    base_idx = batch_idx * (num_channels * spatial_size) + channel_idx * spatial_size
    
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        idx = base_idx + offsets
        
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        sum_squares += tl.sum(vals * vals)
    
    instance_std = tl.sqrt(sum_squares / spatial_size + eps)
    
    # Load v1 and gamma parameters
    v1 = tl.load(v1_ptr + channel_idx)
    gamma = tl.load(gamma_ptr + channel_idx)
    
    denominator = tl.maximum(v1 * instance_std, eps)
    
    # Apply normalization
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        idx = base_idx + offsets
        
        vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
        output = (vals / denominator) + gamma
        
        tl.store(output_ptr + idx, output, mask=mask)

def evonorm_b0(input_tensor, v1, gamma, eps=1e-5):
    """EvoNorm-B0 from NeurIPS 2020 paper"""
    batch_size, num_channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 256
    total_instances = batch_size * num_channels
    grid = (total_instances,)
    
    evonorm_b0_kernel[grid](
        input_tensor, output, v1, gamma,
        batch_size, num_channels, spatial_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output