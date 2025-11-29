%%writefile triton_utils.py

import triton_fix

import torch
import time

class CudaTimer:
    """GPU-accurate timing using CUDA events"""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0
    
    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()
    
    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
    
    def get_elapsed_ms(self):
        return self.elapsed_time

def initialize_weights(shape, scale=1.0):
    """Initialize weights with He initialization"""
    weights = torch.randn(shape, dtype=torch.float32) * scale
    return weights

def zero_tensor(shape):
    """Create zero-initialized tensor"""
    return torch.zeros(shape, dtype=torch.float32)