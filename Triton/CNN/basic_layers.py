%%writefile basic_layers.py

import triton_fix

import torch
import torch.nn.functional as F

def conv2d_layer(input_tensor, weight, bias, padding=1, stride=1):
    """Convolution using PyTorch (Triton doesn't need custom conv)"""
    return F.conv2d(input_tensor, weight, bias, stride=stride, padding=padding)

def maxpool2d_layer(input_tensor, kernel_size=2, stride=2):
    """Max pooling using PyTorch"""
    return F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)

def fc_layer(input_tensor, weight, bias):
    """Fully connected layer"""
    return F.linear(input_tensor, weight, bias)

def softmax_layer(input_tensor, dim=-1):
    """Softmax activation"""
    return F.softmax(input_tensor, dim=dim)

def cross_entropy_loss(logits, labels):
    """Cross entropy loss"""
    return F.cross_entropy(logits, labels)