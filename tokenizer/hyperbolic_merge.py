#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperbolic Tokenizer implementation.

This module implements the core hyperbolic tokenization algorithm, which performs
merges based on hyperbolic distance in the embedding space.
"""

import torch
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional, Set, Union
from tqdm import tqdm
import logging
import sys
import os

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.lorentz_model import distance, exp_map, log_map, project_to_hyperboloid

logger = logging.getLogger(__name__)


class HyperbolicTokenizer:
    """
    Tokenizer that uses hyperbolic geometry to guide subword merges.
    
    Instead of frequency-based merges (as in BPE) or likelihood-based merges (as in Unigram),
    this tokenizer uses hyperbolic distance between token embeddings to determine merge candidates.
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        curvature: float = 1.0,
        merge_threshold: float = 0.1,
        lr: float = 1e-3,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            curvature: Curvature parameter of the hyperbolic space
            merge_threshold: Initial threshold for merging tokens
            lr: Learning rate for the RSGD optimizer
            device: Device to use for computation
        """
        self.vocab = vocab
        self.embeddings = embeddings
        self.curvature = curvature
        self.merge_threshold = merge_threshold
        self.lr = lr
        
        # Set device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        self.embeddings = self.embeddings.to(self.device)
        
        # Build token to index mapping
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Initialize merge stats
        self.merge_history = []
        
    def _compute_pairwise_distances(self) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances between all token embeddings.
        
        Returns:
            Tensor of shape (len(vocab), len(vocab)) with pairwise distances
        """
        n = len(self.vocab)
        distances = torch.zeros((n, n), device=self.device)
        
        # Compute distances in batches to avoid memory issues
        batch_size = 128
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                
                # Compute distances between embeddings[i:end_i] and embeddings[j:end_j]
                emb_i = self.embeddings[i:end_i].unsqueeze(1)  # (batch, 1, d+1)
                emb_j = self.embeddings[j:end_j].unsqueeze(0)  # (1, batch, d+1)
                
                dist_batch = distance(emb_i, emb_j, self.curvature)  # (batch_i, batch_j)
                distances[i:end_i, j:end_j] = dist_batch
        
        return distances
    
    def _find_merge_candidates(self) -> List[Tuple[int, int, float]]:
        """
        Find candidate token pairs for merging based on hyperbolic distance.
        
        Returns:
            List of tuples (i, j, dist) representing merge candidates
        """
        # Compute pairwise distances
        distances = self._compute_pairwise_distances()
        
        # Create a mask for valid merges (only adjacent tokens in the vocabulary)
        n = len(self.vocab)
        valid_mask = torch.zeros((n, n), dtype=torch.bool, device=self.device)
        
        # Find token pairs that could form valid merges
        for i, token_i in enumerate(self.vocab):
            for j, token_j in enumerate(self.vocab):
                # Check if these tokens can be merged (they form a continuous sequence)
                merged = token_i + token_j
                valid_mask[i, j] = self._is_valid_merge(token_i, token_j)
        
        # Apply mask and find candidates below threshold
        masked_distances = torch.where(valid_mask, distances, torch.tensor(float('inf'), device=self.device))
        candidates = torch.nonzero(masked_distances < self.merge_threshold, as_tuple=True)
        
        # Convert to list of tuples with distances
        return [(i.item(), j.item(), distances[i, j].item()) for i, j in zip(*candidates)]
    
    def _is_valid_merge(self, token_i: str, token_j: str) -> bool:
        """
        Check if two tokens form a valid merge.
        
        Args:
            token_i: First token
            token_j: Second token
            
        Returns:
            Whether the merge is valid
        """
        # In a real implementation, this would check if the tokens
        # frequently appear together in the training data
        # For simplicity, we'll just check if they could form a continuous sequence
        return True  # Simplified for this implementation
    
    def _merge_tokens(self, i: int, j: int) -> None:
        """
        Merge two tokens and update the vocabulary and embeddings.
        
        Args:
            i: Index of first token
            j: Index of second token
        """
        token_i = self.vocab[i]
        token_j = self.vocab[j]
        merged_token = token_i + token_j
        
        # Create a new embedding for the merged token
        # In hyperbolic space, we use a weighted midpoint operation
        weight_i = len(token_i) / (len(token_i) + len(token_j))
        weight_j = len(token_j) / (len(token_i) + len(token_j))
        
        # Compute the weighted midpoint in tangent space then project back
        xi = self.embeddings[i].unsqueeze(0)
        xj = self.embeddings[j].unsqueeze(0)
        
        # Log map to tangent space at xi
        v_j_at_i = log_map(xi, xj, self.curvature)
        
        # Scale by weight
        v_scaled = v_j_at_i * weight_j
        
        # Exponential map back to manifold
        x_merged = exp_map(xi, v_scaled, self.curvature)[0]
        
        # Normalize to ensure it stays on the manifold
        x_merged = project_to_hyperboloid(x_merged, self.curvature)
        
        # Add the merged token to vocabulary and embeddings
        self.vocab.append(merged_token)
        self.token2idx[merged_token] = len(self.vocab) - 1
        
        # Extend embeddings tensor
        new_embeddings = torch.nn.Parameter(
            torch.cat([self.embeddings, x_merged.unsqueeze(0)], dim=0)
        )
        self.embeddings = new_embeddings.to(self.device)
        
        # Record the merge in history
        self.merge_history.append((token_i, token_j, merged_token))
        
    def optimize_merges(self, steps: int = 10000, log_every: int = 1000) -> None:
        """
        Perform iterative merge optimization to build the vocabulary.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
        """
        pbar = tqdm(range(steps), desc="Optimizing merges")
        
        for step in pbar:
            # Find merge candidates
            candidates = self._find_merge_candidates()
            
            if not candidates:
                logger.info(f"No more merge candidates found after {step} steps")
                break
            
            # Sort candidates by distance (ascending)
            candidates.sort(key=lambda x: x[2])
            
            # Select the best candidate
            i, j, dist = candidates[0]
            
            # Perform the merge
            self._merge_tokens(i, j)
            
            # Log progress
            if (step + 1) % log_every == 0:
                logger.info(f"Step {step+1}: merged '{self.vocab[i]}' + '{self.vocab[j]}' -> "
                            f"'{self.vocab[-1]}' (dist: {dist:.4f})")
                logger.info(f"Vocabulary size: {len(self.vocab)}")
                
            # Update progress bar
            pbar.set_postfix({
                "vocab_size": len(self.vocab),
                "best_dist": dist,
                "threshold": self.merge_threshold
            })
            
            # Adaptive threshold adjustment (optional)
            if step > 0 and step % 1000 == 0:
                self.merge_threshold *= 1.05  # Gradually increase threshold
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string into subword tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # This is a simplified tokenization algorithm
        # A real implementation would use the merge rules more efficiently
        
        # Start with character-level tokenization
        tokens = list(text)
        
        # Apply merges iteratively
        for _ in range(len(self.merge_history)):
            i = 0
            while i < len(tokens) - 1:
                pair = tokens[i] + tokens[i + 1]
                if pair in self.token2idx:
                    tokens[i] = pair
                    tokens.pop(i + 1)
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        return [self.token2idx.get(token, self.token2idx.get("<unk>", 3)) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices back to text.
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text
        """
        return "".join(self.vocab[idx] for idx in indices)
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            path: Directory path to save the tokenizer
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.vocab, f)
        
        # Save embeddings
        torch.save(self.embeddings.detach().cpu(), f"{path}/embeddings.pt")
        
        # Save merge history
        with open(f"{path}/merges.json", "w") as f:
            json.dump(self.merge_history, f)
        
        # Save config
        config = {
            "curvature": self.curvature,
            "merge_threshold": self.merge_threshold,
            "embedding_dim": self.embeddings.size(1) - 1
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "HyperbolicTokenizer":
        """
        Load a tokenizer from disk.
        
        Args:
            path: Directory path to load the tokenizer from
            device: Device to load the tokenizer onto
            
        Returns:
            Loaded tokenizer
        """
        # Load vocabulary
        with open(f"{path}/vocab.json", "r") as f:
            vocab = json.load(f)
        
        # Load embeddings
        embeddings = torch.load(f"{path}/embeddings.pt")
        
        # Load config
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab=vocab,
            embeddings=torch.nn.Parameter(embeddings),
            curvature=config["curvature"],
            merge_threshold=config["merge_threshold"],
            device=device
        )
        
        # Load merge history
        with open(f"{path}/merges.json", "r") as f:
            tokenizer.merge_history = json.load(f)
            
        return tokenizer
