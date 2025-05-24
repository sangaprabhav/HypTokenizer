#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperbolic contrastive loss for multimodal learning.

This module implements contrastive loss functions in hyperbolic space,
specifically designed for cross-modal retrieval tasks.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from ..embedding.lorentz_model import distance


def hyperbolic_contrastive_loss(
    z_text: torch.Tensor,
    z_img: torch.Tensor,
    temp: float = 0.07,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute hyperbolic contrastive loss between text and image embeddings.
    
    Args:
        z_text: Text embeddings in Lorentz model, shape (batch_size, d+1)
        z_img: Image embeddings in Lorentz model, shape (batch_size, d+1)
        temp: Temperature parameter
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        Contrastive loss value
    """
    batch_size = z_text.size(0)
    
    # Compute pairwise distances
    text_to_img_dist = torch.zeros((batch_size, batch_size), device=z_text.device)
    
    for i in range(batch_size):
        text_to_img_dist[i] = distance(
            z_text[i].unsqueeze(0).expand(batch_size, -1),
            z_img,
            c=1.0
        )
    
    # Convert distances to similarities (smaller distance = higher similarity)
    similarities = -text_to_img_dist / temp
    
    # Labels are the diagonal indices (positive pairs)
    labels = torch.arange(batch_size, device=z_text.device)
    
    # Compute cross-entropy loss in both directions
    loss_t2i = F.cross_entropy(similarities, labels, reduction=reduction)
    loss_i2t = F.cross_entropy(similarities.t(), labels, reduction=reduction)
    
    # Average the two directions
    loss = (loss_t2i + loss_i2t) / 2.0
    
    return loss


def hyperbolic_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute hyperbolic triplet loss.
    
    Args:
        anchor: Anchor embeddings in Lorentz model, shape (batch_size, d+1)
        positive: Positive embeddings in Lorentz model, shape (batch_size, d+1)
        negative: Negative embeddings in Lorentz model, shape (batch_size, d+1)
        margin: Margin parameter
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        Triplet loss value
    """
    # Compute distances
    d_pos = distance(anchor, positive, c=1.0)
    d_neg = distance(anchor, negative, c=1.0)
    
    # Compute loss
    losses = F.relu(d_pos - d_neg + margin)
    
    # Apply reduction
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:  # none
        return losses


class HyperbolicInfoNCE(torch.nn.Module):
    """
    Hyperbolic InfoNCE loss for contrastive learning in hyperbolic space.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hyperbolic InfoNCE loss.
        
        Args:
            z1: First set of embeddings in Lorentz model, shape (batch_size, d+1)
            z2: Second set of embeddings in Lorentz model, shape (batch_size, d+1)
            
        Returns:
            InfoNCE loss value
        """
        return hyperbolic_contrastive_loss(z1, z2, temp=self.temperature)


class MultimodalHyperbolicModel(torch.nn.Module):
    """
    Two-tower model for multimodal contrastive learning in hyperbolic space.
    """
    
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        projection_dim: int = 64,
        curvature: float = 1.0
    ):
        """
        Initialize the multimodal model.
        
        Args:
            text_encoder: Text encoder model
            image_encoder: Image encoder model
            projection_dim: Dimension of the projection space
            curvature: Curvature parameter
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        # Get encoder output dimensions
        self.text_dim = self._get_encoder_dim(text_encoder)
        self.image_dim = self._get_encoder_dim(image_encoder)
        
        # Projection heads
        self.text_projector = torch.nn.Sequential(
            torch.nn.Linear(self.text_dim, projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_dim, projection_dim + 1)  # +1 for Lorentz model
        )
        
        self.image_projector = torch.nn.Sequential(
            torch.nn.Linear(self.image_dim, projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_dim, projection_dim + 1)  # +1 for Lorentz model
        )
        
        self.curvature = curvature
        
    def _get_encoder_dim(self, encoder: torch.nn.Module) -> int:
        """Helper method to get the output dimension of an encoder."""
        # This is a placeholder - in practice, you would inspect the encoder architecture
        return 768  # Assuming BERT/ViT-like models
        
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project a point to the hyperboloid manifold."""
        x_spatial = x[:, 1:]
        x_spatial_norm = torch.norm(x_spatial, dim=1, keepdim=True)
        x_0 = torch.sqrt(1.0 + self.curvature * x_spatial_norm * x_spatial_norm)
        return torch.cat([x_0, x[:, 1:]], dim=1)
    
    def encode_text(self, text_inputs: dict) -> torch.Tensor:
        """
        Encode text inputs to hyperbolic space.
        
        Args:
            text_inputs: Text inputs
            
        Returns:
            Text embeddings in Lorentz model
        """
        text_features = self.text_encoder(**text_inputs)
        if isinstance(text_features, tuple):
            text_features = text_features[0]  # Take the main output if it's a tuple
            
        # If text_features is a dict with 'pooler_output' (like BERT)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
            
        text_projected = self.text_projector(text_features)
        return self.project_to_hyperboloid(text_projected)
    
    def encode_image(self, image_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode image inputs to hyperbolic space.
        
        Args:
            image_inputs: Image inputs
            
        Returns:
            Image embeddings in Lorentz model
        """
        image_features = self.image_encoder(image_inputs)
        if isinstance(image_features, tuple):
            image_features = image_features[0]  # Take the main output if it's a tuple
            
        # If image_features is a dict with 'pooler_output'
        if hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
            
        image_projected = self.image_projector(image_features)
        return self.project_to_hyperboloid(image_projected)
    
    def forward(
        self,
        text_inputs: dict,
        image_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multimodal model.
        
        Args:
            text_inputs: Text inputs
            image_inputs: Image inputs
            
        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        text_embeddings = self.encode_text(text_inputs)
        image_embeddings = self.encode_image(image_inputs)
        
        return text_embeddings, image_embeddings
