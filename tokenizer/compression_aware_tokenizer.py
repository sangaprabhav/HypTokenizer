#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compression-Aware Hyperbolic Tokenizer implementation.

This module implements a hyperbolic tokenizer that optimizes for compression ratio
while maintaining token quality, evaluating merge candidates based on both
hyperbolic distance and compression efficiency.
"""

import torch
import numpy as np
import logging
import os
import sys
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
from tqdm import tqdm

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import distance

logger = logging.getLogger(__name__)


class CompressionAwareTokenizer(HyperbolicTokenizer):
    """
    Tokenizer that optimizes for compression ratio while maintaining token quality.
    
    This tokenizer evaluates merge candidates based on both hyperbolic distance
    and compression efficiency, prioritizing merges that effectively reduce the
    number of tokens needed to encode text.
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        corpus_sample: Optional[List[str]] = None,
        compression_weight: float = 0.7,
        distance_weight: float = 0.3,
        sample_size: int = 100,
        curvature: float = 1.0,
        merge_threshold: float = 0.1,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True
    ):
        """
        Initialize the compression-aware hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            corpus_sample: Sample of text for evaluating compression efficiency
            compression_weight: Weight for compression ratio in scoring (0.0-1.0)
            distance_weight: Weight for hyperbolic distance in scoring (0.0-1.0)
            sample_size: Number of candidates to evaluate for compression
            curvature: Curvature parameter of the hyperbolic space
            merge_threshold: Threshold for considering merge candidates
            lr: Learning rate for embedding updates
            device: Device to use for computation
            max_vocab_size: Maximum vocabulary size
            use_approximate_search: Whether to use approximate search for large vocabularies
        """
        super().__init__(
            vocab=vocab,
            embeddings=embeddings,
            curvature=curvature,
            merge_threshold=merge_threshold,
            lr=lr,
            device=device,
            max_vocab_size=max_vocab_size,
            use_approximate_search=use_approximate_search
        )
        
        # Store parameters
        self.compression_weight = compression_weight
        self.distance_weight = distance_weight
        self.sample_size = sample_size
        
        # Store corpus sample for compression evaluation
        self.corpus_sample = corpus_sample or []
        
        # Cache for tokenization results
        self.tokenize_cache = {}
    
    def _tokenize_with_vocab(self, text: str, vocab: List[str]) -> List[str]:
        """
        Tokenize text using a specific vocabulary.
        
        This is a simple greedy longest-match tokenization for evaluation purposes.
        
        Args:
            text: Text to tokenize
            vocab: Vocabulary to use for tokenization
            
        Returns:
            List of tokens
        """
        # Sort vocab by length for greedy longest match
        sorted_vocab = sorted(vocab, key=len, reverse=True)
        
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted_vocab:
                if text[i:].startswith(token):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # If no match found, add character as token
                tokens.append(text[i])
                i += 1
        
        return tokens
    
    def _compression_aware_scoring(self, candidates: List[Tuple[int, int, float]]) -> List[float]:
        """
        Score candidates based on compression efficiency.
        
        Args:
            candidates: List of (i, j, distance) merge candidates
            
        Returns:
            List of scores for each candidate (higher is better)
        """
        if not self.corpus_sample:
            # If no corpus sample, fall back to distance-based scoring
            return [1.0 / (1.0 + dist) for _, _, dist in candidates]
        
        # Compute original tokenization length once
        cache_key = "original"
        if cache_key not in self.tokenize_cache:
            self.tokenize_cache[cache_key] = sum(len(self.tokenize(text)) for text in self.corpus_sample)
        original_tokens = self.tokenize_cache[cache_key]
        
        scores = []
        
        # Take a sample of candidates to evaluate
        sample_size = min(self.sample_size, len(candidates))
        candidates_to_evaluate = candidates[:sample_size]
        
        for i, j, dist in candidates_to_evaluate:
            # Simulate merge
            merged_token = self.vocab[i] + self.vocab[j]
            temp_vocab = self.vocab.copy()
            temp_vocab.append(merged_token)
            
            # Create a simple set of merge rules for tokenization
            merge_rules = {f"{self.vocab[i]}{self.vocab[j]}": merged_token}
            
            # Compute compression ratio
            merged_tokens = 0
            for text in self.corpus_sample:
                # Use cached tokenization if available
                cache_key = f"merge_{i}_{j}_{text[:20]}"
                if cache_key not in self.tokenize_cache:
                    # Simple tokenization for evaluation
                    tokens = self._tokenize_with_vocab(text, temp_vocab)
                    self.tokenize_cache[cache_key] = len(tokens)
                
                merged_tokens += self.tokenize_cache[cache_key]
            
            # Avoid division by zero
            if merged_tokens == 0:
                compression_ratio = 1.0
            else:
                compression_ratio = original_tokens / merged_tokens
            
            # Distance score (lower distance = higher score)
            distance_score = 1.0 / (1.0 + dist)
            
            # Combined score with weighting
            score = (self.compression_weight * compression_ratio + 
                     self.distance_weight * distance_score)
            
            scores.append(score)
        
        # For candidates we didn't evaluate, use distance-based scoring
        if sample_size < len(candidates):
            scores.extend([1.0 / (1.0 + dist) for _, _, dist in candidates[sample_size:]])
        
        return scores
    
    def _find_merge_candidates(self) -> List[Tuple[int, int, float]]:
        """
        Find potential merge candidates and score them with compression awareness.
        
        Returns:
            List of (i, j, score) tuples for merge candidates, sorted by score (higher is better)
        """
        # Get candidates based on distance threshold
        candidates = super()._find_merge_candidates()
        
        if not candidates:
            return []
        
        # Score candidates with compression awareness
        scores = self._compression_aware_scoring(candidates)
        
        # Create new candidates with scores as the distance (will be sorted)
        # We use negative scores because sorting is ascending, but higher scores are better
        scored_candidates = [(i, j, -score) for (i, j, _), score in zip(candidates, scores)]
        
        # Sort by score (ascending, but scores are negative so best first)
        scored_candidates.sort(key=lambda x: x[2])
        
        return scored_candidates
    
    def optimize_merges(self, 
                        steps: int = 10000, 
                        log_every: int = 1000,
                        corpus_sample: Optional[List[str]] = None) -> None:
        """
        Perform iterative merge optimization with compression awareness.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            corpus_sample: Optional corpus sample for compression evaluation
        """
        # Update corpus sample if provided
        if corpus_sample:
            self.corpus_sample = corpus_sample
            # Clear cache when corpus changes
            self.tokenize_cache = {}
        
        # Perform optimization
        from tqdm import tqdm
        
        pbar = tqdm(range(steps), desc="Optimizing merges")
        
        for step in pbar:
            # Find merge candidates with compression-aware scoring
            candidates = self._find_merge_candidates()
            
            if not candidates:
                logger.info(f"No more merge candidates found after {step} steps")
                break
            
            # Take the best candidate
            i, j, neg_score = candidates[0]
            score = -neg_score  # Convert back to positive score
            
            # Perform the merge
            self._merge_tokens(i, j)
            
            # Clear part of the cache that might be affected by this merge
            keys_to_remove = [k for k in self.tokenize_cache if k.startswith("merge_")]
            for k in keys_to_remove:
                self.tokenize_cache.pop(k)
            
            # Update progress
            pbar.set_postfix({
                "vocab_size": self.current_vocab_size,
                "score": f"{score:.4f}",
                "threshold": f"{self.merge_threshold:.4f}"
            })
            
            # Log progress
            if (step + 1) % log_every == 0:
                logger.info(
                    f"Step {step+1}: merged '{self.vocab[i]}' + '{self.vocab[j]}' -> '{self.vocab[-1]}' "
                    f"(score: {score:.4f})"
                )
            
            # Adaptive threshold adjustment if needed
            if step > 0 and step % 1000 == 0:
                self.merge_threshold *= 1.1
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            path: Directory path to save the tokenizer
        """
        # Call parent implementation first
        super().save(path)
        
        # Save compression-specific parameters
        import json
        
        compression_config = {
            "compression_weight": self.compression_weight,
            "distance_weight": self.distance_weight,
            "sample_size": self.sample_size
        }
        
        with open(f"{path}/compression_config.json", "w") as f:
            json.dump(compression_config, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'CompressionAwareTokenizer':
        """
        Load a tokenizer from disk.
        
        Args:
            path: Directory path to load the tokenizer from
            device: Device to load the tokenizer onto
            
        Returns:
            Loaded tokenizer
        """
        import json
        
        # Load base tokenizer first
        tokenizer = super().load(path, device)
        
        # Convert to CompressionAwareTokenizer
        compression_config = {}
        try:
            with open(f"{path}/compression_config.json", "r") as f:
                compression_config = json.load(f)
        except FileNotFoundError:
            logger.warning("Compression config file not found, using defaults")
        
        compression_tokenizer = cls(
            vocab=tokenizer.vocab,
            embeddings=tokenizer.embeddings,
            curvature=tokenizer.curvature,
            merge_threshold=tokenizer.merge_threshold,
            device=device if device is not None else tokenizer.device,
            compression_weight=compression_config.get("compression_weight", 0.7),
            distance_weight=compression_config.get("distance_weight", 0.3),
            sample_size=compression_config.get("sample_size", 100)
        )
        
        # Copy over attributes
        compression_tokenizer.merge_history = tokenizer.merge_history
        compression_tokenizer.current_vocab_size = tokenizer.current_vocab_size
        
        return compression_tokenizer
