%%writefile mnist_loader.py

import triton_fix

import struct
import numpy as np
import torch

def read_idx_images(filepath):
    """Load MNIST images from IDX format"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
        
    return images, num_images, num_rows, num_cols

def read_idx_labels(filepath):
    """Load MNIST labels from IDX format"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels, num_labels

def convert_images_to_float(images):
    """Convert uint8 images to normalized float32"""
    # Normalize: (pixel/255 - 0.1307) / 0.3081
    images_float = images.astype(np.float32) / 255.0
    images_float = (images_float - 0.1307) / 0.3081
    return images_float

def load_mnist_dataset(train_images_path, train_labels_path, 
                       test_images_path, test_labels_path):
    """Load complete MNIST dataset"""
    train_images, num_train, img_h, img_w = read_idx_images(train_images_path)
    train_labels, _ = read_idx_labels(train_labels_path)
    test_images, num_test, _, _ = read_idx_images(test_images_path)
    test_labels, _ = read_idx_labels(test_labels_path)
    
    # Convert to float and normalize
    train_images = convert_images_to_float(train_images)
    test_images = convert_images_to_float(test_images)
    
    return train_images, train_labels, test_images, test_labels