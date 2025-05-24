#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test MPS availability and functionality.
"""

import torch
import time

def main():
    # Check if MPS is available
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create tensors on device
    start_time = time.time()
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    # Perform matrix multiplication
    for _ in range(100):
        c = torch.matmul(a, b)
    
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.4f} seconds")
    
    # Try hyperboloid operations
    print("\nTesting hyperboloid operations:")
    # Initialize in Lorentz model
    x = torch.zeros(51, device=device)
    x[0] = 1.0  # Origin in Lorentz model
    
    y = torch.zeros(51, device=device)
    y[0] = 1.1
    y[1] = 0.458
    
    # Compute Minkowski dot product
    dot = x[0] * y[0] - torch.sum(x[1:] * y[1:])
    print(f"Minkowski dot product: {dot.item()}")
    
if __name__ == "__main__":
    main()
