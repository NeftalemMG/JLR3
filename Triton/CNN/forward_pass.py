%%writefile forward_pass.py

import triton_fix

import torch
from basic_layers import conv2d_layer, maxpool2d_layer, fc_layer, softmax_layer
from activation_kernels import relu_activation, gelu_activation, swish_activation
from normalization_kernels import layernorm, evonorm_b0
from fused_kernels import gelu_layernorm_fused, swish_layernorm_fused

def forward_mode0(model, input_images):
    """Mode 0: ReLU + LayerNorm (NON-FUSED - 2 separate operations)"""
    # Convolution
    conv_out = conv2d_layer(input_images, model.conv_weight, model.conv_bias, 
                           padding=model.config.conv_padding, stride=model.config.conv_stride)
    
    # ReLU activation (SEPARATE)
    B, C, H, W = conv_out.shape
    activated = relu_activation(conv_out.view(-1)).view(B, C, H, W)
    
    # LayerNorm (SEPARATE)
    # Reshape for layernorm: (B, C*H*W)
    B, C, H, W = activated.shape
    activated_flat = activated.view(B, -1)
    gamma_flat = model.layernorm_gamma.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    beta_flat = model.layernorm_beta.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    
    normalized_flat = layernorm(activated_flat, gamma_flat, beta_flat, model.config.layernorm_epsilon)
    normalized = normalized_flat.view(B, C, H, W)
    
    # Max pooling
    pooled = maxpool2d_layer(normalized, kernel_size=2, stride=2)
    
    # Fully connected
    fc_input = pooled.view(B, -1)
    logits = fc_layer(fc_input, model.fc_weight, model.fc_bias)
    
    # Softmax
    probabilities = softmax_layer(logits, dim=1)
    
    return logits, probabilities

def forward_mode1(model, input_images):
    """Mode 1: GELU + LayerNorm (FUSED - 1 kernel)"""
    # Convolution
    conv_out = conv2d_layer(input_images, model.conv_weight, model.conv_bias,
                           padding=model.config.conv_padding, stride=model.config.conv_stride)
    
    # GELU + LayerNorm FUSED (ONE operation)
    B, C, H, W = conv_out.shape
    conv_flat = conv_out.view(B, -1)
    gamma_flat = model.layernorm_gamma.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    beta_flat = model.layernorm_beta.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    
    normalized_flat = gelu_layernorm_fused(conv_flat, gamma_flat, beta_flat, model.config.layernorm_epsilon)
    normalized = normalized_flat.view(B, C, H, W)
    
    # Max pooling
    pooled = maxpool2d_layer(normalized, kernel_size=2, stride=2)
    
    # Fully connected
    fc_input = pooled.view(B, -1)
    logits = fc_layer(fc_input, model.fc_weight, model.fc_bias)
    
    # Softmax
    probabilities = softmax_layer(logits, dim=1)
    
    return logits, probabilities

def forward_mode2(model, input_images):
    """Mode 2: EvoNorm-B0 (FUSED - inherently combined)"""
    # Convolution
    conv_out = conv2d_layer(input_images, model.conv_weight, model.conv_bias,
                           padding=model.config.conv_padding, stride=model.config.conv_stride)
    
    # EvoNorm-B0 FUSED
    normalized = evonorm_b0(conv_out, model.evonorm_v1, model.evonorm_gamma, model.config.layernorm_epsilon)
    
    # Max pooling
    pooled = maxpool2d_layer(normalized, kernel_size=2, stride=2)
    
    # Fully connected
    B = pooled.shape[0]
    fc_input = pooled.view(B, -1)
    logits = fc_layer(fc_input, model.fc_weight, model.fc_bias)
    
    # Softmax
    probabilities = softmax_layer(logits, dim=1)
    
    return logits, probabilities

def forward_mode3(model, input_images):
    """Mode 3: Swish + LayerNorm (FUSED - 1 kernel)"""
    # Convolution
    conv_out = conv2d_layer(input_images, model.conv_weight, model.conv_bias,
                           padding=model.config.conv_padding, stride=model.config.conv_stride)
    
    # Swish + LayerNorm FUSED (ONE operation)
    B, C, H, W = conv_out.shape
    conv_flat = conv_out.view(B, -1)
    gamma_flat = model.layernorm_gamma.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    beta_flat = model.layernorm_beta.unsqueeze(1).unsqueeze(2).expand(-1, H, W).reshape(-1)
    
    normalized_flat = swish_layernorm_fused(conv_flat, gamma_flat, beta_flat, 
                                            model.config.swish_beta, model.config.layernorm_epsilon)
    normalized = normalized_flat.view(B, C, H, W)
    
    # Max pooling
    pooled = maxpool2d_layer(normalized, kernel_size=2, stride=2)
    
    # Fully connected
    fc_input = pooled.view(B, -1)
    logits = fc_layer(fc_input, model.fc_weight, model.fc_bias)
    
    # Softmax
    probabilities = softmax_layer(logits, dim=1)
    
    return logits, probabilities

def forward_pass(model, input_images, mode):
    """Forward pass dispatcher"""
    if mode == 0:
        return forward_mode0(model, input_images)
    elif mode == 1:
        return forward_mode1(model, input_images)
    elif mode == 2:
        return forward_mode2(model, input_images)
    elif mode == 3:
        return forward_mode3(model, input_images)
    else:
        raise ValueError(f"Invalid mode: {mode}")