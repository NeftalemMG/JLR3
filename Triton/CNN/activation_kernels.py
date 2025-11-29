%%writefile activation_kernels.py

import triton_fix

import torch
import triton
import triton.language as tl

# RELU Activation

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_activation(input_tensor):
    """ReLU activation: max(0, x)"""
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    relu_kernel[grid](
        input_tensor, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# GELU Activation

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    
    x_cubed = x * x * x
    inner = x + coeff * x_cubed
    tanh_arg = sqrt_2_over_pi * inner
    tanh_val = tl.math.tanh(tanh_arg)
    output = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu_activation(input_tensor):
    """GELU activation"""
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](
        input_tensor, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Swish Activation
@triton.jit
def swish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(beta * x)
    sigmoid = 1.0 / (1.0 + tl.math.exp(-beta * x))
    output = x * sigmoid
    
    tl.store(output_ptr + offsets, output, mask=mask)

def swish_activation(input_tensor, beta=1.0):
    """Swish activation: x * sigmoid(beta * x)"""
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    swish_kernel[grid](
        input_tensor, output, n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output