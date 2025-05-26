#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frequency-Aware Hyperbolic Tokenizer implementation.

This module extends the HyperbolicTokenizer to incorporate frequency information
from a corpus, combining hyperbolic distance with frequency statistics and
semantic coherence for better linguistically valid merge decisions.
"""

import torch
import numpy as np
import logging
import os
import sys
from typing import List, Dict, Tuple, Optional, Set, Union
from tqdm import tqdm
import time

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import distance, log_map, exp_map

logger = logging.getLogger(__name__)


class FrequencyAwareHyperbolicTokenizer(HyperbolicTokenizer):
    """
    Tokenizer that combines hyperbolic geometry with frequency information.
    
    Extends the HyperbolicTokenizer to incorporate frequency information from a corpus,
    balancing geometric distances with linguistic co-occurrence patterns.
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        corpus_path: Optional[str] = None,
        alpha: float = 0.4,  # Weight for distance score
        beta: float = 0.4,   # Weight for frequency score
        gamma: float = 0.2,  # Weight for semantic coherence
        curvature: float = 1.0,
        merge_threshold: float = 1.0,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True
    ):
        """
        Initialize the frequency-aware hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            corpus_path: Path to corpus file for computing frequencies
            alpha: Weight for hyperbolic distance score
            beta: Weight for frequency score
            gamma: Weight for semantic coherence score
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
        
        # Store hyperparameters for scoring
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize pair frequencies
        self.pair_frequencies: Dict[Tuple[str, str], int] = {}
        
        # Compute frequencies if corpus path is provided
        if corpus_path:
            self._compute_pair_frequencies(corpus_path)
    
    def _compute_pair_frequencies(self, corpus_path: str) -> None:
        """
        Compute token pair frequencies from corpus.
        
        Args:
            corpus_path: Path to corpus file
        """
        logger.info("Computing pair frequencies from corpus...")
        total_pairs = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Computing frequencies"):
                tokens = self.tokenize(line.strip())
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    self.pair_frequencies[pair] = self.pair_frequencies.get(pair, 0) + 1
                    total_pairs += 1
        
        logger.info(f"Computed frequencies for {len(self.pair_frequencies)} unique token pairs "
                   f"from {total_pairs} total pairs")
    
    def _compute_semantic_coherence(self, i: int, j: int) -> float:
        """
        Compute semantic coherence score for merging tokens i and j.
        
        This measures how well the merged token fits in the embedding space
        using distances to neighboring tokens.
        
        Args:
            i: Index of first token
            j: Index of second token
            
        Returns:
            Semantic coherence score (higher is better)
        """
        with torch.no_grad():
            # Get embeddings
            emb_i = self.embeddings[i].unsqueeze(0)
            emb_j = self.embeddings[j].unsqueeze(0)
            
            # Compute midpoint in tangent space (this simulates the merged embedding)
            token_i, token_j = self.vocab[i], self.vocab[j]
            weight_i = len(token_i) / (len(token_i) + len(token_j))
            weight_j = len(token_j) / (len(token_i) + len(token_j))
            
            # Compute the midpoint in tangent space
            v_j_at_i = log_map(emb_i, emb_j, self.curvature)
            v_scaled = v_j_at_i * weight_j
            merged_emb = exp_map(emb_i, v_scaled, self.curvature)
            
            # Sample a small set of tokens to compare against
            sample_size = min(50, self.current_vocab_size)
            indices = torch.randperm(self.current_vocab_size)[:sample_size]
            
            # Compute distances from merged token to sampled tokens
            distances = []
            for idx in indices:
                if idx != i and idx != j:  # Skip the tokens being merged
                    emb_k = self.embeddings[idx].unsqueeze(0)
                    dist = distance(merged_emb, emb_k, self.curvature).item()
                    distances.append(dist)
            
            if not distances:
                return 0.0
                
            # Coherence is inversely related to the average distance to other tokens
            # We want merged tokens that maintain good separation from others
            avg_dist = np.mean(distances)
            
            # Transform to a 0-1 score (lower distances → higher score)
            # Using a sigmoid-like function centered around the merge threshold
            coherence = 1.0 / (1.0 + np.exp(avg_dist - self.merge_threshold))
            
            return coherence
    
    def _score_merge_candidate(self, i: int, j: int, dist: float) -> float:
        """
        Score merge candidate combining distance, frequency, and semantic coherence.
        
        Args:
            i: Index of first token
            j: Index of second token
            dist: Hyperbolic distance between token embeddings
            
        Returns:
            Combined score (higher is better)
        """
        token_i, token_j = self.vocab[i], self.vocab[j]
        
        # Hyperbolic distance score (lower distance → higher score)
        dist_score = 1.0 / (1.0 + dist)
        
        # Frequency score
        pair_freq = self.pair_frequencies.get((token_i, token_j), 0)
        freq_score = np.log1p(pair_freq)  # Log to handle skewed distributions
        
        # Normalize frequency score to 0-1 range
        max_freq = max(self.pair_frequencies.values()) if self.pair_frequencies else 1
        freq_score = freq_score / np.log1p(max_freq) if max_freq > 0 else 0
        
        # Semantic coherence score
        semantic_score = self._compute_semantic_coherence(i, j)
        
        # Combined score with hyperparameter weights
        score = self.alpha * dist_score + self.beta * freq_score + self.gamma * semantic_score
        
        return score
    
    def _find_merge_candidates(self) -> List[Tuple[int, int, float]]:
        """
        Find potential merge candidates based on hyperbolic distance and frequency.
        
        Overrides the base method to incorporate frequency information and semantic coherence.
        
        Returns:
            List of (i, j, score) tuples for merge candidates, sorted by score (higher is better)
        """
        # Get candidates based on distance threshold
        candidates = super()._find_merge_candidates()
        
        # Skip scoring if no candidates or no frequency data
        if not candidates or (self.beta > 0 and not self.pair_frequencies):
            return candidates
        
        # Score each candidate
        scored_candidates = []
        for i, j, dist in candidates:
            # Skip if tokens can't be merged
            if not self._is_valid_merge(self.vocab[i], self.vocab[j]):
                continue
                
            # Compute combined score
            score = self._score_merge_candidate(i, j, dist)
            
            # Store as (i, j, score) - we use negative score because we sort ascending
            # but want higher scores to be better
            scored_candidates.append((i, j, -score))
        
        # Sort by score (ascending, but scores are negative so best first)
        scored_candidates.sort(key=lambda x: x[2])
        
        return scored_candidates
    
    def optimize_merges(self, 
                        steps: int = 10000, 
                        log_every: int = 1000,
                        parallel_eval: bool = True, 
                        sample_ratio: float = 1.0,
                        corpus_path: Optional[str] = None) -> None:
        """
        Perform iterative merge optimization to build the vocabulary.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            parallel_eval: Whether to evaluate candidates in parallel
            sample_ratio: Ratio of candidates to sample for evaluation (0.0-1.0)
            corpus_path: Path to corpus file for updating frequencies
        """
        # Update frequencies if new corpus provided
        if corpus_path:
            self._compute_pair_frequencies(corpus_path)
            
        # Perform optimization with our improved scoring
        from tqdm import tqdm
        
        pbar = tqdm(range(steps), desc="Optimizing merges")
        no_candidate_count = 0
        
        for step in pbar:
            start_time = time.time()
            
            # Find merge candidates
            candidates = self._find_merge_candidates()
            
            # Log statistics
            if step % log_every == 0:
                logger.info(f"Step {step}: vocab_size={self.current_vocab_size}")
                logger.info(f"  Merge candidates: {len(candidates)}")
                logger.info(f"  Merge threshold: {self.merge_threshold:.6f}")
            
            if not candidates:
                no_candidate_count += 1
                
                if no_candidate_count > 5:
                    # Increase threshold if we can't find candidates
                    self.merge_threshold *= 1.5
                    logger.info(f"No candidates found. Increasing threshold to {self.merge_threshold:.6f}")
                    no_candidate_count = 0
                    continue
                elif no_candidate_count > 10:
                    logger.info(f"No more merge candidates found after {step} steps")
                    break
                continue
            else:
                no_candidate_count = 0
            
            # Pick the best candidate and perform the merge
            i, j, score = candidates[0]
            self._merge_tokens(i, j)
            
            # Candidates are sorted by negative score, so we negate again to display
            pbar.set_postfix({
                "vocab_size": self.current_vocab_size,
                "score": f"{-score:.4f}",
                "threshold": f"{self.merge_threshold:.4f}",
                "time": f"{time.time() - start_time:.2f}s"
            })
            
            # Log progress
            if (step + 1) % log_every == 0:
                token_i, token_j = self.vocab[i], self.vocab[j]
                merged_token = self.vocab[-1]
                logger.info(
                    f"Step {step+1}: merged '{token_i}' + '{token_j}' -> '{merged_token}' "
                    f"(score: {-score:.4f})"
                )
            
            # Adaptive threshold adjustment
            if step > 0 and step % 1000 == 0:
                self.merge_threshold *= 1.1  # Gradual increase

    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk, including frequency information.
        
        Args:
            path: Directory path to save the tokenizer
        """
        # Call parent implementation first
        super().save(path)
        
        # Save frequency information
        import json
        
        # Convert tuple keys to lists for JSON serialization
        frequencies_json = {f"{k[0]}|{k[1]}": v for k, v in self.pair_frequencies.items()}
        
        with open(f"{path}/frequencies.json", "w") as f:
            json.dump(frequencies_json, f)
        
        # Save hyperparameters
        hyperparams = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }
        
        with open(f"{path}/freq_hyperparams.json", "w") as f:
            json.dump(hyperparams, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'FrequencyAwareHyperbolicTokenizer':
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
        
        # Convert to FrequencyAwareHyperbolicTokenizer
        freq_tokenizer = cls(
            vocab=tokenizer.vocab,
            embeddings=tokenizer.embeddings,
            curvature=tokenizer.curvature,
            merge_threshold=tokenizer.merge_threshold,
            device=device if device is not None else tokenizer.device
        )
        
        # Copy over attributes
        freq_tokenizer.merge_history = tokenizer.merge_history
        freq_tokenizer.current_vocab_size = tokenizer.current_vocab_size
        
        # Load hyperparameters if available
        try:
            with open(f"{path}/freq_hyperparams.json", "r") as f:
                hyperparams = json.load(f)
                freq_tokenizer.alpha = hyperparams.get("alpha", 0.4)
                freq_tokenizer.beta = hyperparams.get("beta", 0.4)
                freq_tokenizer.gamma = hyperparams.get("gamma", 0.2)
        except FileNotFoundError:
            logger.warning("Hyperparameters file not found, using defaults")
        
        # Load frequencies if available
        try:
            with open(f"{path}/frequencies.json", "r") as f:
                frequencies_json = json.load(f)
                
                # Convert serialized keys back to tuples
                freq_tokenizer.pair_frequencies = {
                    tuple(k.split("|")): v for k, v in frequencies_json.items()
                }
        except FileNotFoundError:
            logger.warning("Frequencies file not found")
        
        return freq_tokenizer
