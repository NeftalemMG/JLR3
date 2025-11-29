%%writefile training_ops.py

import triton_fix

import torch
from basic_layers import cross_entropy_loss
from forward_pass import forward_pass

def train_step(model, batch_images, batch_labels, mode, optimizer):
    """Single training step"""
    optimizer.zero_grad()
    
    # Forward pass
    logits, probabilities = forward_pass(model, batch_images, mode)
    
    # Compute loss
    loss = cross_entropy_loss(logits, batch_labels)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    return loss.item()

def evaluate_model(model, test_images, test_labels, batch_size, mode):
    """Evaluate model accuracy"""
    num_samples = test_images.shape[0]
    num_batches = num_samples // batch_size
    total_correct = 0
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            
            batch_images = test_images[batch_start:batch_end]
            batch_labels = test_labels[batch_start:batch_end]
            
            # Forward pass
            logits, probabilities = forward_pass(model, batch_images, mode)
            
            # Compute accuracy
            predictions = torch.argmax(probabilities, dim=1)
            correct = (predictions == batch_labels).sum().item()
            total_correct += correct
    
    accuracy = 100.0 * total_correct / (num_batches * batch_size)
    return accuracy