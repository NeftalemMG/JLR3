%%writefile model.py

import triton_fix

import torch
import torch.nn as nn
from triton_utils import initialize_weights, zero_tensor

class NetworkConfig:
    """Configuration for CNN"""
    def __init__(self):
        self.num_classes = 10
        self.image_size = 28
        self.num_input_channels = 1
        self.num_conv_filters = 16
        self.kernel_size = 3
        self.conv_padding = 1
        self.conv_stride = 1
        self.conv_output_height = 28
        self.conv_output_width = 28
        self.pool_output_height = 14
        self.pool_output_width = 14
        self.fc_input_dimension = self.num_conv_filters * self.pool_output_height * self.pool_output_width
        self.batch_size = 64
        self.num_epochs = 3
        self.learning_rate = 0.01
        self.layernorm_epsilon = 1e-5
        self.swish_beta = 1.0

class CNNModel:
    """CNN Model with different fusion modes"""
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # Initialize weights
        conv_scale = (2.0 / (config.num_input_channels * config.kernel_size * config.kernel_size)) ** 0.5
        self.conv_weight = initialize_weights(
            (config.num_conv_filters, config.num_input_channels, config.kernel_size, config.kernel_size),
            conv_scale
        ).to(device)
        self.conv_bias = zero_tensor((config.num_conv_filters,)).to(device)
        
        fc_scale = (2.0 / config.fc_input_dimension) ** 0.5
        self.fc_weight = initialize_weights(
            (config.num_classes, config.fc_input_dimension),
            fc_scale
        ).to(device)
        self.fc_bias = zero_tensor((config.num_classes,)).to(device)
        
        # Normalization parameters
        self.layernorm_gamma = torch.ones(config.num_conv_filters, device=device)
        self.layernorm_beta = torch.zeros(config.num_conv_filters, device=device)
        
        self.evonorm_v1 = torch.ones(config.num_conv_filters, device=device)
        self.evonorm_gamma = torch.zeros(config.num_conv_filters, device=device)
        
        # Make parameters trainable (for FC layer only in this simple version)
        self.fc_weight.requires_grad = True
        self.fc_bias.requires_grad = True
    
    def get_trainable_params(self):
        """Return list of trainable parameters"""
        return [self.fc_weight, self.fc_bias]