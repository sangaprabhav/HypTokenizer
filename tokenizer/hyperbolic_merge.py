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
import warnings

# Setup logger
logger = logging.getLogger(__name__)

# Optional imports for fast nearest neighbor search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Falling back to brute force search for large vocabularies.")

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.lorentz_model import distance, batch_distance, exp_map, log_map, project_to_hyperboloid

# Handle PyTorch 2.0+ optimizations based on device type
try:
    import torch._dynamo
    
    # Check if torch.compile is available
    TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')
    
    # Function to conditionally apply torch.compile based on device type
    def maybe_compile(func):
        """
        Apply torch.compile only on CUDA devices, skip on MPS and CPU.
        
        Args:
            func: The function to compile
            
        Returns:
            Either the compiled function (CUDA) or the original function (MPS, CPU)
        """
        # Get the current device
        device_type = None
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
        
        # Only apply torch.compile on CUDA devices
        if TORCH_COMPILE_AVAILABLE and device_type == 'cuda':
            try:
                return torch.compile(func)
            except Exception as e:
                logger.warning(f"Failed to compile function: {e}")
                return func
        else:
            # On MPS or CPU, return the original function
            return func
    
    # Apply the conditional compilation
    distance_compiled = maybe_compile(distance)
    batch_distance_compiled = maybe_compile(batch_distance)
    
    # Set a flag to indicate if compilation is actually used
    USING_COMPILED = TORCH_COMPILE_AVAILABLE and torch.cuda.is_available()
    if USING_COMPILED:
        logger.info("Using torch.compile for acceleration on CUDA device")
    elif TORCH_COMPILE_AVAILABLE:
        logger.info("torch.compile is available but not used (non-CUDA device detected)")
    else:
        logger.info("torch.compile is not available (PyTorch < 2.0)")
        
except ImportError:
    # Fallback if torch._dynamo is not available
    TORCH_COMPILE_AVAILABLE = False
    USING_COMPILED = False
    distance_compiled = distance
    batch_distance_compiled = batch_distance


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
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True
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
            max_vocab_size: Maximum vocabulary size to pre-allocate
            use_approximate_search: Whether to use approximate nearest neighbor search for large vocabularies
        """
        # Set device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
        # Save initial parameters
        self.vocab = vocab.copy()  # Make a copy to avoid modifying the input
        self.current_vocab_size = len(vocab)
        self.max_vocab_size = max_vocab_size
        self.curvature = curvature
        self.merge_threshold = merge_threshold
        self.lr = lr
        self.use_approximate_search = use_approximate_search and FAISS_AVAILABLE
        
        # Pre-allocate embeddings tensor for efficiency
        embedding_dim = embeddings.size(1)
        full_embeddings = torch.zeros(
            (max_vocab_size, embedding_dim),
            dtype=embeddings.dtype,
            device=self.device
        )
        # Copy initial embeddings to pre-allocated tensor
        full_embeddings[:self.current_vocab_size] = embeddings.to(self.device)
        self.embeddings = torch.nn.Parameter(full_embeddings)
        
        # Build token to index mapping
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Initialize merge stats
        self.merge_history = []
        
        # Initialize FAISS index if available and requested
        self.index = None
        if self.use_approximate_search:
            self._init_faiss_index()
        
    def _compute_pairwise_distances(self) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances between all token embeddings.
        
        Returns:
            Tensor of shape (len(vocab), len(vocab)) with pairwise distances
        """
        n = self.current_vocab_size
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
        Find potential merge candidates based on hyperbolic distance.
        
        Returns:
            List of (i, j, distance) tuples for merge candidates
        """
        candidates = []
        n = self.current_vocab_size
        
        # For very large vocabularies, use approximate nearest neighbor search
        if self.use_approximate_search and n > 10000:
            # Update the FAISS index with current embeddings
            self._update_faiss_index()
            
            # For each token, find its nearest neighbors
            with torch.no_grad():
                # Convert to Klein model for querying
                klein_embeddings = self.embeddings[:n, 1:] / self.embeddings[:n, 0:1]
                query_data = klein_embeddings.detach().cpu().numpy().astype('float32')
                
                # Number of neighbors to search for each point
                k = min(100, n)  # Adjust based on expected merges per token
                
                # Search for nearest neighbors
                distances, indices = self.index.search(query_data, k)
                
                # Process results to find merge candidates
                for i in range(n):
                    for j_idx in range(1, k):  # Skip the first result (self)
                        j = indices[i, j_idx]
                        
                        # Skip invalid indices
                        if j >= n or j <= i:
                            continue
                        
                        # Compute the exact hyperbolic distance for validation
                        # Use compiled version if available
                        if TORCH_COMPILE_AVAILABLE:
                            dist = distance_compiled(
                                self.embeddings[i].unsqueeze(0),
                                self.embeddings[j].unsqueeze(0),
                                self.curvature
                            ).item()
                        else:
                            dist = distance(
                                self.embeddings[i].unsqueeze(0),
                                self.embeddings[j].unsqueeze(0),
                                self.curvature
                            ).item()
                        
                        if dist < self.merge_threshold:
                            candidates.append((i, j, dist))
                            
        # For medium-sized vocabularies, use batch computation
        elif n > 100:  
            with torch.no_grad():
                # Only use the active part of the pre-allocated tensor
                active_embeddings = self.embeddings[:n]
                
                # Compute all pairwise distances using batch_distance (compiled if available)
                if TORCH_COMPILE_AVAILABLE:
                    all_dists = batch_distance_compiled(active_embeddings, active_embeddings, self.curvature)
                else:
                    all_dists = batch_distance(active_embeddings, active_embeddings, self.curvature)
                
                # Create mask to avoid self-comparisons and only consider upper triangular matrix (i < j)
                mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=self.device), diagonal=1)
                
                # Find candidates below threshold
                valid_pairs = (all_dists < self.merge_threshold) & mask
                candidate_indices = valid_pairs.nonzero(as_tuple=True)
                
                # Extract candidates
                for idx in range(len(candidate_indices[0])):
                    i = candidate_indices[0][idx].item()
                    j = candidate_indices[1][idx].item()
                    candidates.append((i, j, all_dists[i, j].item()))
        else:
            # Original implementation for small vocabularies
            for i in range(n):
                for j in range(i + 1, n):
                    # Use compiled version if available
                    if TORCH_COMPILE_AVAILABLE:
                        dist = distance_compiled(
                            self.embeddings[i].unsqueeze(0),
                            self.embeddings[j].unsqueeze(0),
                            self.curvature
                        ).item()
                    else:
                        dist = distance(
                            self.embeddings[i].unsqueeze(0),
                            self.embeddings[j].unsqueeze(0),
                            self.curvature
                        ).item()
                    
                    if dist < self.merge_threshold:
                        candidates.append((i, j, dist))
    
        return candidates
    
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
        
        # Check if we've reached maximum vocabulary size
        if self.current_vocab_size >= self.max_vocab_size:
            raise ValueError(f"Maximum vocabulary size {self.max_vocab_size} reached. Cannot merge more tokens.")
        
        # Add the merged token to vocabulary
        self.vocab.append(merged_token)
        self.token2idx[merged_token] = self.current_vocab_size
        
        # Use the pre-allocated space in the embeddings tensor
        self.embeddings.data[self.current_vocab_size] = x_merged
        self.current_vocab_size += 1
        
        # Record the merge in history
        self.merge_history.append((token_i, token_j, merged_token))
        
    def optimize_merges(self, steps: int = 10000, log_every: int = 1000, parallel_eval: bool = True, sample_ratio: float = 1.0) -> None:
        """
        Perform iterative merge optimization to build the vocabulary.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            parallel_eval: Whether to evaluate candidates in parallel
            sample_ratio: Ratio of candidates to sample for evaluation (0.0-1.0)
        """
        pbar = tqdm(range(steps), desc="Optimizing merges")
        
        for step in pbar:
            # Find merge candidates based on hyperbolic distance
            candidates = self._find_merge_candidates()
            
            if not candidates:
                logger.info("No more merge candidates found. Stopping.")
                break
                
            # Sort by distance (ascending)
            candidates.sort(key=lambda x: x[2])
            
            # Use sampling to reduce number of candidates if requested
            if sample_ratio < 1.0:
                sample_size = max(1, int(len(candidates) * sample_ratio))
                candidates = candidates[:sample_size]
            
            # Evaluate multiple candidates in parallel if requested
            if parallel_eval and len(candidates) > 1 and torch.cuda.is_available():
                # Get top-k candidates
                top_k = min(100, len(candidates))
                top_candidates = candidates[:top_k]
                
                # Evaluate top candidates and select the best
                best_candidate = self._evaluate_candidates_parallel(top_candidates)
                i, j, dist = best_candidate
            else:
                # Get the best candidate
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
            
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string into subword tokens.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        # Build merge rules dictionary for O(1) lookup if not already built
        if not hasattr(self, '_merge_rules'):
            self._merge_rules = {}
            for old1, old2, new in self.merge_history:
                self._merge_rules[(old1, old2)] = new
        
        tokens = list(text)
        changed = True
        
        # Apply merges until no more can be applied
        while changed:
            changed = False
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merge_rules:
                    tokens[i] = self._merge_rules[pair]
                    tokens.pop(i + 1)
                    changed = True
                else:
                    i += 1
    
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into token indices.
        
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
        
        # Save only the active embeddings
        active_embeddings = self.embeddings[:self.current_vocab_size].detach().cpu()
        torch.save(active_embeddings, f"{path}/embeddings.pt")
        
        # Save merge history
        with open(f"{path}/merges.json", "w") as f:
            json.dump(self.merge_history, f)
        
        # Save config
        config = {
            "curvature": self.curvature,
            "merge_threshold": self.merge_threshold,
            "embedding_dim": self.embeddings.size(1) - 1,
            "max_vocab_size": self.max_vocab_size,
            "use_approximate_search": self.use_approximate_search
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
        
        # Get additional parameters from config or use defaults
        max_vocab_size = config.get("max_vocab_size", 100000)
        use_approximate_search = config.get("use_approximate_search", True)
        
        # Create tokenizer
        tokenizer = cls(
            vocab=vocab,
            embeddings=torch.nn.Parameter(embeddings),
            curvature=config["curvature"],
            merge_threshold=config["merge_threshold"],
            device=device,
            max_vocab_size=max_vocab_size,
            use_approximate_search=use_approximate_search
        )
        
        # Load merge history
        with open(f"{path}/merges.json", "r") as f:
            tokenizer.merge_history = json.load(f)
        
        # Make sure current_vocab_size is set correctly
        tokenizer.current_vocab_size = len(tokenizer.vocab)
        
        return tokenizer
    
    def _evaluate_candidates_parallel(self, candidates: List[Tuple[int, int, float]]) -> Tuple[int, int, float]:
        """
        Evaluate multiple merge candidates in parallel and pick the best one.
        
        Args:
            candidates: List of candidate tuples (i, j, distance)
            
        Returns:
            The best candidate (i, j, distance)
        """
        # Create a list to store simulated merged embeddings
        simulated_embeddings = []
        
        # Process each candidate
        with torch.no_grad():
            for i, j, _ in candidates:
                token_i = self.vocab[i]
                token_j = self.vocab[j]
                
                # Compute the weights for the merge
                weight_i = len(token_i) / (len(token_i) + len(token_j))
                weight_j = len(token_j) / (len(token_i) + len(token_j))
                
                # Get embeddings
                xi = self.embeddings[i].unsqueeze(0)
                xj = self.embeddings[j].unsqueeze(0)
                
                # Compute weighted midpoint in tangent space
                v_j_at_i = log_map(xi, xj, self.curvature)
                v_scaled = v_j_at_i * weight_j
                x_merged = exp_map(xi, v_scaled, self.curvature)[0]
                x_merged = project_to_hyperboloid(x_merged, self.curvature)
                
                # Store the simulated merged embedding
                simulated_embeddings.append(x_merged)
        
        # Evaluate all candidates at once (could implement a perplexity or hierarchy quality metric here)
        # For now, just return the candidate with smallest distance
        return candidates[0]
    
    def _init_faiss_index(self) -> None:
        """
        Initialize FAISS index for fast nearest neighbor search.
        """
        if not FAISS_AVAILABLE:
            return
        
        # Get the embedding dimension (minus the time dimension for Lorentz model)
        dim = self.embeddings.size(1) - 1
        
        # Create a FAISS index - we'll use the spatial dimensions only (not the time dimension)
        # For hyperbolic space, we use L2 distance in Klein model as an approximation
        self.index = faiss.IndexFlatL2(dim)

    def _update_faiss_index(self) -> None:
        """
        Update FAISS index with current embeddings.
        """
        if not self.use_approximate_search or self.index is None:
            return
        
        # Convert Lorentz embeddings to Klein model for approximate search
        # We use the Klein model because it's in Euclidean space where L2 distance approximates hyperbolic distance
        with torch.no_grad():
            # We need to use just the spatial dimensions (1:) and normalize by time dimension (0)
            klein_embeddings = self.embeddings[:self.current_vocab_size, 1:] / self.embeddings[:self.current_vocab_size, 0:1]
        
        # Convert to numpy for FAISS
        index_data = klein_embeddings.detach().cpu().numpy().astype('float32')
        
        # Clear and re-add all vectors
        self.index.reset()
        self.index.add(index_data)
