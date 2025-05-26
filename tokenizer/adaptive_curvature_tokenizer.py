#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Curvature Hyperbolic Tokenizer implementation.

This module extends the HyperbolicTokenizer to dynamically optimize the
curvature parameter during training, better capturing hierarchical structure.
"""

import torch
import numpy as np
import logging
import os
import sys
from typing import List, Dict, Tuple, Optional, Set, Any
from tqdm import tqdm
import time

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import (
    distance, batch_distance, exp_map, log_map, project_to_hyperboloid, 
    poincare_to_lorentz, lorentz_to_poincare
)

logger = logging.getLogger(__name__)


class AdaptiveCurvatureTokenizer(HyperbolicTokenizer):
    """
    Hyperbolic tokenizer with adaptive curvature parameter.
    
    This tokenizer dynamically optimizes the curvature parameter during training,
    allowing it to better capture hierarchical relationships in the token space.
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        curvature: float = 1.0,
        merge_threshold: float = 0.1,
        lr: float = 1e-3,
        curvature_lr: float = 0.01,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True,
        hierarchy_weight: float = 1.0,
        distortion_weight: float = 0.1,
        optimize_freq: int = 100  # How often to optimize curvature
    ):
        """
        Initialize the adaptive curvature hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            curvature: Initial curvature parameter of the hyperbolic space
            merge_threshold: Threshold for considering merge candidates
            lr: Learning rate for embedding updates
            curvature_lr: Learning rate for curvature optimization
            device: Device to use for computation
            max_vocab_size: Maximum vocabulary size
            use_approximate_search: Whether to use approximate search for large vocabularies
            hierarchy_weight: Weight for hierarchy preservation loss
            distortion_weight: Weight for distortion loss
            optimize_freq: How often to optimize curvature (in merge steps)
        """
        # Initialize with parent's constructor, but replace curvature with a parameter
        super().__init__(
            vocab=vocab,
            embeddings=embeddings,
            # Use a dummy value here, we'll override with a parameter
            curvature=1.0,
            merge_threshold=merge_threshold,
            lr=lr,
            device=device,
            max_vocab_size=max_vocab_size,
            use_approximate_search=use_approximate_search
        )
        
        # Replace static curvature with a trainable parameter
        self.curvature = torch.nn.Parameter(torch.tensor(curvature, device=self.device))
        
        # Create optimizer for curvature
        self.curvature_optimizer = torch.optim.Adam([self.curvature], lr=curvature_lr)
        
        # Store additional parameters
        self.hierarchy_weight = hierarchy_weight
        self.distortion_weight = distortion_weight
        self.optimize_freq = optimize_freq
        
        # Project embeddings to ensure they lie on the hyperboloid with current curvature
        with torch.no_grad():
            self.embeddings.data = project_to_hyperboloid(self.embeddings.data, self.curvature.item())
        
        # Track merge pairs for hierarchy information
        self.merge_pairs: List[Tuple[int, int]] = []
    
    def _compute_hierarchy_preservation_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on how well hierarchical relationships are preserved.
        
        Uses merge history to define hierarchical relationships - tokens that were
        merged should be closer to each other than to other tokens.
        
        Args:
            embeddings: Current token embeddings
            
        Returns:
            Hierarchy preservation loss
        """
        if not self.merge_pairs:
            # If we don't have merge history yet, return zero loss
            return torch.tensor(0.0, device=self.device)
        
        loss = torch.tensor(0.0, device=self.device)
        
        # Sample a subset of merge pairs to make computation tractable
        num_pairs = min(len(self.merge_pairs), 100)
        pairs_to_use = self.merge_pairs[-num_pairs:]
        
        for i, j in pairs_to_use:
            # Skip if tokens no longer exist (have been merged)
            if i >= len(embeddings) or j >= len(embeddings):
                continue
                
            # Get embeddings
            emb_i = embeddings[i].unsqueeze(0)
            emb_j = embeddings[j].unsqueeze(0)
            
            # Compute distance between merged tokens
            pair_dist = distance(emb_i, emb_j, self.curvature)
            
            # Sample other tokens to compare against
            num_samples = min(10, len(embeddings) - 2)
            sample_indices = torch.randperm(len(embeddings))[:num_samples]
            sample_indices = sample_indices[~torch.isin(sample_indices, torch.tensor([i, j], device=self.device))]
            
            if len(sample_indices) == 0:
                continue
                
            # Compute distances to other tokens
            other_dists_i = torch.stack([
                distance(emb_i, embeddings[k].unsqueeze(0), self.curvature)
                for k in sample_indices
            ])
            
            other_dists_j = torch.stack([
                distance(emb_j, embeddings[k].unsqueeze(0), self.curvature)
                for k in sample_indices
            ])
            
            # Hierarchical loss: merged tokens should be closer to each other 
            # than to other tokens
            margin = 0.1
            hier_loss_i = torch.relu(pair_dist - other_dists_i + margin).mean()
            hier_loss_j = torch.relu(pair_dist - other_dists_j + margin).mean()
            
            loss = loss + hier_loss_i + hier_loss_j
        
        # Normalize by number of pairs
        if num_pairs > 0:
            loss = loss / (2 * num_pairs)
        
        return loss
    
    def _compute_distortion_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute distortion loss to avoid excessive curvature.
        
        This loss ensures that the embeddings don't collapse to a single point
        or get pushed too far apart.
        
        Args:
            embeddings: Current token embeddings
            
        Returns:
            Distortion loss
        """
        # Sample pairs to compute average distance
        n = len(embeddings)
        num_samples = min(500, n * (n - 1) // 2)
        
        dists = []
        for _ in range(num_samples):
            i, j = torch.randint(0, n, (2,))
            if i != j:
                dist = distance(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                    self.curvature
                )
                dists.append(dist)
        
        if not dists:
            return torch.tensor(0.0, device=self.device)
            
        # Compute mean and variance of distances
        dists_tensor = torch.stack(dists)
        mean_dist = dists_tensor.mean()
        var_dist = dists_tensor.var()
        
        # We want to avoid very small mean distances (collapse)
        # and very large variance (excessive distortion)
        collapse_loss = torch.exp(-10 * mean_dist)  # Penalize small mean distance
        variance_loss = var_dist  # Penalize high variance
        
        # Combined distortion loss
        distortion_loss = collapse_loss + 0.1 * variance_loss
        
        return distortion_loss
    
    def _optimize_curvature(self, embeddings: torch.Tensor) -> None:
        """
        Optimize curvature based on hierarchy preservation and distortion.
        
        Args:
            embeddings: Current token embeddings
        """
        # Compute hierarchy preservation loss
        hierarchy_loss = self._compute_hierarchy_preservation_loss(embeddings)
        
        # Compute distortion loss
        distortion_loss = self._compute_distortion_loss(embeddings)
        
        # Combined loss
        loss = self.hierarchy_weight * hierarchy_loss + self.distortion_weight * distortion_loss
        
        # Optimize
        self.curvature_optimizer.zero_grad()
        loss.backward()
        self.curvature_optimizer.step()
        
        # Ensure curvature stays positive
        with torch.no_grad():
            self.curvature.clamp_(min=0.1, max=10.0)
        
        logger.info(f"Optimized curvature: {self.curvature.item():.4f}, "
                   f"Loss: {loss.item():.4f} (H: {hierarchy_loss.item():.4f}, D: {distortion_loss.item():.4f})")
    
    def _project_embeddings(self) -> None:
        """
        Project embeddings to the hyperboloid with current curvature.
        """
        with torch.no_grad():
            self.embeddings.data = project_to_hyperboloid(self.embeddings.data, self.curvature.item())
    
    def _merge_tokens(self, i: int, j: int) -> None:
        """
        Merge two tokens and update the vocabulary and embeddings.
        
        Overridden to track merge pairs for hierarchy preservation.
        
        Args:
            i: Index of first token
            j: Index of second token
        """
        # Track merge pair for hierarchy preservation
        self.merge_pairs.append((i, j))
        
        # Call parent implementation
        super()._merge_tokens(i, j)
    
    def optimize_merges(self, 
                        steps: int = 10000, 
                        log_every: int = 1000,
                        parallel_eval: bool = True, 
                        sample_ratio: float = 1.0) -> None:
        """
        Perform iterative merge optimization with adaptive curvature.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            parallel_eval: Whether to evaluate candidates in parallel
            sample_ratio: Ratio of candidates to sample for evaluation (0.0-1.0)
        """
        from tqdm import tqdm
        
        pbar = tqdm(range(steps), desc="Optimizing merges")
        
        for step in pbar:
            # Find merge candidates using current curvature
            candidates = self._find_merge_candidates()
            
            if not candidates:
                logger.info(f"No more merge candidates found after {step} steps")
                break
            
            # Periodically optimize curvature
            if step > 0 and step % self.optimize_freq == 0:
                # Detach embeddings for optimization to avoid affecting the merge
                self._optimize_curvature(self.embeddings.detach())
                
                # Project embeddings to ensure they lie on the hyperboloid with updated curvature
                self._project_embeddings()
            
            # Sample candidates if needed
            if sample_ratio < 1.0:
                sample_size = max(1, int(len(candidates) * sample_ratio))
                candidates = candidates[:sample_size]
            
            # Evaluate candidates in parallel if requested
            if parallel_eval and len(candidates) > 1:
                best_candidate = self._evaluate_candidates_parallel(candidates)
                i, j, dist = best_candidate
            else:
                # Take the best candidate
                i, j, dist = candidates[0]
            
            # Perform the merge
            self._merge_tokens(i, j)
            
            # Update progress
            pbar.set_postfix({
                "vocab_size": self.current_vocab_size,
                "distance": f"{dist:.4f}",
                "curvature": f"{self.curvature.item():.4f}",
                "threshold": f"{self.merge_threshold:.4f}"
            })
            
            # Log progress
            if (step + 1) % log_every == 0:
                logger.info(
                    f"Step {step+1}: merged '{self.vocab[i]}' + '{self.vocab[j]}' -> '{self.vocab[-1]}' "
                    f"(dist: {dist:.4f}, curvature: {self.curvature.item():.4f})"
                )
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            path: Directory path to save the tokenizer
        """
        import json
        import os
        
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.vocab, f)
        
        # Save embeddings
        torch.save(self.embeddings, f"{path}/embeddings.pt")
        
        # Save curvature as a separate tensor
        torch.save(self.curvature, f"{path}/curvature.pt")
        
        # Save merge history
        with open(f"{path}/merges.json", "w") as f:
            json.dump(self.merge_history, f)
        
        # Save merge pairs for hierarchy
        torch.save(self.merge_pairs, f"{path}/merge_pairs.pt")
        
        # Save configuration
        config = {
            "curvature": self.curvature.item(),
            "merge_threshold": self.merge_threshold,
            "max_vocab_size": self.max_vocab_size,
            "use_approximate_search": self.use_approximate_search,
            "hierarchy_weight": self.hierarchy_weight,
            "distortion_weight": self.distortion_weight,
            "optimize_freq": self.optimize_freq
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'AdaptiveCurvatureTokenizer':
        """
        Load a tokenizer from disk.
        
        Args:
            path: Directory path to load the tokenizer from
            device: Device to load the tokenizer onto
            
        Returns:
            Loaded tokenizer
        """
        import json
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else 
                                 "cpu")
        
        # Load vocabulary
        with open(f"{path}/vocab.json", "r") as f:
            vocab = json.load(f)
        
        # Load embeddings
        embeddings = torch.load(f"{path}/embeddings.pt", map_location=device)
        
        # Load curvature
        curvature_tensor = torch.load(f"{path}/curvature.pt", map_location=device)
        curvature = curvature_tensor.item()
        
        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab=vocab,
            embeddings=embeddings,
            curvature=curvature,
            merge_threshold=config["merge_threshold"],
            device=device,
            max_vocab_size=config.get("max_vocab_size", 100000),
            use_approximate_search=config.get("use_approximate_search", True),
            hierarchy_weight=config.get("hierarchy_weight", 1.0),
            distortion_weight=config.get("distortion_weight", 0.1),
            optimize_freq=config.get("optimize_freq", 100)
        )
        
        # Load merge history
        with open(f"{path}/merges.json", "r") as f:
            tokenizer.merge_history = json.load(f)
        
        # Load merge pairs if available
        try:
            tokenizer.merge_pairs = torch.load(f"{path}/merge_pairs.pt", map_location=device)
        except FileNotFoundError:
            logger.warning("Merge pairs file not found")
        
        # Make sure current_vocab_size is set correctly
        tokenizer.current_vocab_size = len(tokenizer.vocab)
        
        return tokenizer
