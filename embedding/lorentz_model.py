#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lorentz model for hyperbolic embeddings.

Implements the core operations in the Lorentz model of hyperbolic space.
"""

import torch
from typing import Tuple, Optional


def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Minkowski dot product for Lorentz model.
    
    Args:
        x: Tensor of shape (..., d+1)
        y: Tensor of shape (..., d+1)
    
    Returns:
        Minkowski dot product between x and y
    """
    return x[..., 0] * y[..., 0] - torch.sum(x[..., 1:] * y[..., 1:], dim=-1)


def minkowski_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Minkowski norm for Lorentz model.
    
    Args:
        x: Tensor of shape (..., d+1)
    
    Returns:
        Minkowski norm of x
    """
    return torch.sqrt(torch.clamp(minkowski_dot(x, x), min=1e-8))


def project_to_hyperboloid(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Project a point to the hyperboloid manifold.
    
    Args:
        x: Tensor of shape (..., d+1)
        c: Curvature parameter
    
    Returns:
        Projected tensor on the hyperboloid
    """
    d = x.size(-1) - 1
    x_norm = torch.norm(x[..., 1:], dim=-1, keepdim=True)
    x_0 = torch.sqrt(1.0 + c * x_norm * x_norm)
    res = torch.cat([x_0, x[..., 1:]], dim=-1)
    return res


def lorentz_to_klein(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Convert from Lorentz to Klein model.
    
    Args:
        x: Tensor of shape (..., d+1) in Lorentz model
        c: Curvature parameter
    
    Returns:
        Tensor in Klein model
    """
    return x[..., 1:] / x[..., 0:1]


def exp_map(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map from tangent space at x to the hyperboloid.
    
    Args:
        x: Base point tensor of shape (..., d+1)
        v: Tangent vector at x, shape (..., d+1)
        c: Curvature parameter
        
    Returns:
        The exponential map result
    """
    v_norm = torch.clamp(torch.sum(v[..., 1:] * v[..., 1:], dim=-1, keepdim=True), min=1e-8)
    v_norm = torch.sqrt(v_norm)
    
    # Handle the case where v_norm is close to zero
    mask = (v_norm < 1e-6).to(v.dtype)
    direction = v / (v_norm + mask)
    direction = mask * torch.zeros_like(direction) + (1 - mask) * direction
    
    return torch.cosh(v_norm) * x + torch.sinh(v_norm) * direction


def log_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map from hyperboloid to tangent space at x.
    
    Args:
        x: Base point tensor of shape (..., d+1)
        y: Target point tensor of shape (..., d+1)
        c: Curvature parameter
        
    Returns:
        The logarithmic map result
    """
    xy_dot = -minkowski_dot(x, y)
    xy_dot = torch.clamp(xy_dot, min=1.0 + 1e-8)  # Ensure it's slightly greater than 1
    
    # Compute the coefficient
    coef = torch.acosh(xy_dot) / torch.sqrt(xy_dot * xy_dot - 1)
    coef = torch.clamp(coef, max=1e4)  # Clip for numerical stability
    
    # Handle the case where coefficient is close to zero or NaN
    mask = ((coef != coef) | (coef > 1e4)).to(coef.dtype)
    coef = mask * torch.ones_like(coef) + (1 - mask) * coef
    
    return coef.unsqueeze(-1) * (y + minkowski_dot(x, y).unsqueeze(-1) * x)


def distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute the hyperbolic distance between points on the hyperboloid.
    
    Args:
        x: Tensor of shape (..., d+1)
        y: Tensor of shape (..., d+1)
        c: Curvature parameter
        
    Returns:
        The hyperbolic distance
    """
    xy_dot = -minkowski_dot(x, y)
    xy_dot = torch.clamp(xy_dot, min=1.0 + 1e-8)  # Ensure it's slightly greater than 1
    # Convert curvature to tensor with the same device as x and y
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    return torch.acosh(xy_dot) / torch.sqrt(c_tensor)


def batch_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute hyperbolic distances between batches of points on the hyperboloid.
    Much faster than individual distance calculations when computing many pairs.
    
    Args:
        x: Tensor of shape (batch_size_1, d+1)
        y: Tensor of shape (batch_size_2, d+1)
        c: Curvature parameter
        
    Returns:
        Tensor of shape (batch_size_1, batch_size_2) with pairwise distances
    """
    # Reshape for broadcasting
    x_reshaped = x.unsqueeze(1)  # Shape: (batch_size_1, 1, d+1)
    y_reshaped = y.unsqueeze(0)  # Shape: (1, batch_size_2, d+1)
    
    # Compute Minkowski dot product for all pairs at once
    # First component multiplication (time-like)
    time_comp = x_reshaped[..., 0] * y_reshaped[..., 0]  # Shape: (batch_size_1, batch_size_2)
    
    # Space-like components dot product
    space_comp = torch.sum(x_reshaped[..., 1:] * y_reshaped[..., 1:], dim=-1)  # Shape: (batch_size_1, batch_size_2)
    
    # Full Minkowski dot product
    mink_dot = time_comp - space_comp  # Shape: (batch_size_1, batch_size_2)
    
    # Negative for distance calculation
    xy_dot = -mink_dot
    
    # Clamp values for numerical stability
    xy_dot = torch.clamp(xy_dot, min=1.0 + 1e-8)
    
    # Convert curvature to tensor
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    
    # Return distances
    return torch.acosh(xy_dot) / torch.sqrt(c_tensor)


def batch_distance_optimized(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Optimized batch distance computation using Einstein summation and in-place operations.
    
    Args:
        x: Tensor of shape (batch_size_1, d+1)
        y: Tensor of shape (batch_size_2, d+1)
        c: Curvature parameter
        
    Returns:
        Tensor of shape (batch_size_1, batch_size_2) with pairwise distances
    """
    # Use Einstein summation for efficiency
    time_comp = torch.einsum('ik,jk->ij', x[..., :1], y[..., :1])
    space_comp = torch.einsum('ik,jk->ij', x[..., 1:], y[..., 1:])
    
    # Compute Minkowski dot product
    mink_dot = time_comp - space_comp
    
    # Negative for distance calculation
    xy_dot = -mink_dot
    
    # Clamp and compute in one operation
    xy_dot_clamped = torch.clamp(xy_dot, min=1.0 + 1e-8)
    
    # Use precomputed square root of curvature for efficiency
    c_sqrt = c ** 0.5
    
    # Return distances
    return torch.acosh(xy_dot_clamped) / c_sqrt


def parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Parallel transport a vector v from point x to point y on the hyperboloid.
    
    Args:
        v: Tangent vector at x, shape (..., d+1)
        x: Source point tensor of shape (..., d+1)
        y: Target point tensor of shape (..., d+1)
        c: Curvature parameter
        
    Returns:
        The parallel transported vector
    """
    xy_dot = -minkowski_dot(x, y).unsqueeze(-1)
    coef = minkowski_dot(y, v).unsqueeze(-1) / (1 - xy_dot)
    return v + coef * (x + y)


def riemannian_gradient(euclidean_grad: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Convert Euclidean gradient to Riemannian gradient on the hyperboloid.
    
    Args:
        euclidean_grad: Euclidean gradient tensor of shape (..., d+1)
        x: Point tensor on the hyperboloid of shape (..., d+1)
        c: Curvature parameter
        
    Returns:
        The Riemannian gradient
    """
    # Project the Euclidean gradient onto the tangent space at x
    return euclidean_grad + minkowski_dot(x, euclidean_grad).unsqueeze(-1) * x
