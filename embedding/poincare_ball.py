#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Poincaré ball model for hyperbolic embeddings.

Implements operations in the Poincaré ball model of hyperbolic space.
"""

import torch
from typing import Tuple, Optional


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean norm.
    
    Args:
        x: Tensor of shape (..., d)
    
    Returns:
        Euclidean norm of x
    """
    return torch.norm(x, dim=-1, keepdim=True)


def mobius_addition(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.
    
    Args:
        x: Tensor of shape (..., d)
        y: Tensor of shape (..., d)
        c: Curvature parameter
    
    Returns:
        Möbius addition result
    """
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy_dot + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
    denom = 1 + 2 * c * xy_dot + c * c * x_norm_sq * y_norm_sq
    
    return num / denom


def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius scalar multiplication in the Poincaré ball.
    
    Args:
        r: Scalar tensor of shape (..., 1)
        x: Tensor of shape (..., d)
        c: Curvature parameter
    
    Returns:
        Möbius scalar multiplication result
    """
    x_norm = norm(x)
    x_norm_clipped = torch.clamp(x_norm, min=1e-8)
    
    return torch.tanh(r * torch.atanh(torch.sqrt(c) * x_norm_clipped)) / \
           (torch.sqrt(c) * x_norm_clipped) * x


def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map from the tangent space at the origin to the Poincaré ball.
    
    Args:
        v: Tangent vector at the origin, shape (..., d)
        c: Curvature parameter
    
    Returns:
        Point in the Poincaré ball
    """
    v_norm = norm(v)
    zeros_mask = (v_norm == 0).to(v.dtype)
    v_norm_clipped = torch.clamp(v_norm, min=1e-8)
    
    return torch.tanh(torch.sqrt(c) * v_norm_clipped) / \
           (torch.sqrt(c) * v_norm_clipped) * v * (1 - zeros_mask) + zeros_mask * v


def log_map_zero(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map from the Poincaré ball to the tangent space at the origin.
    
    Args:
        x: Point in the Poincaré ball, shape (..., d)
        c: Curvature parameter
    
    Returns:
        Tangent vector at the origin
    """
    x_norm = norm(x)
    zeros_mask = (x_norm == 0).to(x.dtype)
    x_norm_clipped = torch.clamp(x_norm, min=1e-8)
    
    return torch.atanh(torch.sqrt(c) * x_norm_clipped) / \
           (torch.sqrt(c) * x_norm_clipped) * x * (1 - zeros_mask) + zeros_mask * x


def distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute the hyperbolic distance between points in the Poincaré ball.
    
    Args:
        x: Tensor of shape (..., d)
        y: Tensor of shape (..., d)
        c: Curvature parameter
    
    Returns:
        The hyperbolic distance
    """
    sqrt_c = torch.sqrt(torch.tensor(c))
    
    # Compute the Möbius addition -x ⊕ y
    neg_x = -x
    neg_x_y = mobius_addition(neg_x, y, c)
    neg_x_y_norm = norm(neg_x_y)
    
    # Distance formula
    return 2 / sqrt_c * torch.atanh(sqrt_c * neg_x_y_norm)


def lorentz_to_poincare(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Convert points from the Lorentz model to the Poincaré ball model.
    
    Args:
        x: Points in the Lorentz model, shape (..., d+1)
        c: Curvature parameter
    
    Returns:
        Points in the Poincaré ball model, shape (..., d)
    """
    return x[..., 1:] / (x[..., 0:1] + 1/torch.sqrt(torch.tensor(c)))


def poincare_to_lorentz(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Convert points from the Poincaré ball model to the Lorentz model.
    
    Args:
        x: Points in the Poincaré ball model, shape (..., d)
        c: Curvature parameter
    
    Returns:
        Points in the Lorentz model, shape (..., d+1)
    """
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    factor = 1.0 / (1 - c * x_norm_sq)
    
    # First component (time-like)
    x0 = factor * (1 + c * x_norm_sq) / (2 * torch.sqrt(torch.tensor(c)))
    
    # Spatial components
    xi = factor * x
    
    return torch.cat([x0, xi], dim=-1)
