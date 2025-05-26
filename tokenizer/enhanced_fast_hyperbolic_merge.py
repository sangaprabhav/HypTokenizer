#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Fast Hyperbolic Tokenizer implementation.

This module implements a state-of-the-art tokenizer that combines:
1. Frequency-aware merging for better linguistic validity
2. Hierarchical merge strategy that respects linguistic structure
3. Adaptive curvature optimization for better hierarchy preservation
4. Compression-aware scoring for efficiency

All these features are integrated with the efficiency of the FastHyperbolicTokenizer.
"""

import torch
import numpy as np
import heapq
import time
import logging
import random
import re
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
import warnings
import os
import sys
from collections import Counter
from tqdm import tqdm

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.fast_hyperbolic_merge import FastHyperbolicTokenizer, MergeCandidate, AdaptiveMergeCache
from embedding.lorentz_model import (
    distance, batch_distance, exp_map, log_map, project_to_hyperboloid,
    poincare_to_lorentz, lorentz_to_poincare
)

# Try to import nltk for morphological analysis
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Morphological filtering will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMergeCandidate(MergeCandidate):
    """Extended merge candidate with additional scores."""
    frequency_score: float = 0.0
    semantic_score: float = 0.0
    compression_score: float = 0.0
    morphology_score: float = 0.0
    combined_score: float = 0.0
    
    def __lt__(self, other):
        # Sort by combined score (lower is better since we negate for heap)
        return self.combined_score < other.combined_score


class EnhancedFastHyperbolicTokenizer(FastHyperbolicTokenizer):
    """
    State-of-the-art hyperbolic tokenizer with all advanced features.
    
    This tokenizer integrates:
    - Fast approximate nearest neighbor search from FastHyperbolicTokenizer
    - Frequency-aware merging for better linguistic validity
    - Hierarchical merge strategy that respects linguistic structure
    - Adaptive curvature optimization for better hierarchy preservation
    - Compression-aware scoring for efficiency
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        # Basic parameters
        curvature: float = 1.0,
        merge_threshold: float = 0.5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True,
        cache_size: int = 10000,
        rebuild_frequency: int = 100,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100,
        
        # Feature flags
        use_frequency_aware: bool = True,
        use_hierarchical: bool = True,
        use_adaptive_curvature: bool = True,
        use_compression_aware: bool = True,
        
        # Frequency-aware parameters
        corpus_path: Optional[str] = None,
        alpha: float = 0.4,  # Weight for distance score
        beta: float = 0.4,   # Weight for frequency score
        gamma: float = 0.2,  # Weight for semantic coherence
        
        # Hierarchical parameters
        language: str = "english",
        
        # Adaptive curvature parameters
        curvature_lr: float = 0.01,
        hierarchy_weight: float = 1.0,
        distortion_weight: float = 0.1,
        optimize_curvature_freq: int = 100,
        
        # Compression-aware parameters
        corpus_sample: Optional[List[str]] = None,
        compression_weight: float = 0.7,
        distance_weight: float = 0.3,
        sample_size: int = 100
    ):
        """
        Initialize the enhanced fast hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model
            
            # Basic parameters
            curvature: Curvature parameter of hyperbolic space (static or initial)
            merge_threshold: Threshold for considering merge candidates
            lr: Learning rate for embedding updates
            device: Device to use for computation
            max_vocab_size: Maximum vocabulary size
            use_approximate_search: Whether to use approximate search for large vocabularies
            cache_size: Size of merge candidate cache
            rebuild_frequency: How often to rebuild the index
            hnsw_m: HNSW parameter for number of connections
            hnsw_ef_construction: HNSW parameter for search during construction
            hnsw_ef_search: HNSW parameter for search during query
            
            # Feature flags
            use_frequency_aware: Whether to use frequency-aware merging
            use_hierarchical: Whether to use hierarchical merge strategy
            use_adaptive_curvature: Whether to use adaptive curvature
            use_compression_aware: Whether to use compression-aware scoring
            
            # Frequency-aware parameters
            corpus_path: Path to corpus file for computing frequencies
            alpha: Weight for distance score
            beta: Weight for frequency score
            gamma: Weight for semantic coherence
            
            # Hierarchical parameters
            language: Language for morphological analysis
            
            # Adaptive curvature parameters
            curvature_lr: Learning rate for curvature optimization
            hierarchy_weight: Weight for hierarchy preservation loss
            distortion_weight: Weight for distortion loss
            optimize_curvature_freq: How often to optimize curvature (in merge steps)
            
            # Compression-aware parameters
            corpus_sample: Sample of text for evaluating compression efficiency
            compression_weight: Weight for compression ratio in scoring
            distance_weight: Weight for hyperbolic distance in scoring
            sample_size: Number of candidates to evaluate for compression
        """
        # Initialize with standard parameters for FastHyperbolicTokenizer
        # (If using adaptive curvature, we'll override this parameter after initialization)
        super().__init__(
            vocab=vocab,
            embeddings=embeddings,
            curvature=curvature,
            merge_threshold=merge_threshold,
            lr=lr,
            device=device,
            max_vocab_size=max_vocab_size,
            use_approximate_search=use_approximate_search,
            cache_size=cache_size,
            rebuild_frequency=rebuild_frequency,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search
        )
        
        # Store feature flags
        self.use_frequency_aware = use_frequency_aware
        self.use_hierarchical = use_hierarchical
        self.use_adaptive_curvature = use_adaptive_curvature
        self.use_compression_aware = use_compression_aware
        
        # Store current merge phase (for hierarchical merging)
        self.current_phase = 1
        
        # Initialize frequency-aware components if enabled
        if self.use_frequency_aware:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.pair_frequencies: Dict[Tuple[str, str], int] = {}
            
            if corpus_path and os.path.exists(corpus_path):
                self._compute_pair_frequencies(corpus_path)
        
        # Initialize hierarchical components if enabled
        if self.use_hierarchical:
            self.language = language
            self.token_frequencies: Dict[str, int] = {}
            self.common_morphemes: Set[str] = set()
            self.common_words: Set[str] = set()
            
            # Initialize NLTK resources if available
            if NLTK_AVAILABLE:
                try:
                    nltk.download('wordnet', quiet=True)
                    self.lemmatizer = WordNetLemmatizer()
                except Exception as e:
                    logger.warning(f"Failed to initialize NLTK resources: {e}")
                    
            # Compute corpus statistics if corpus path provided
            if corpus_path and os.path.exists(corpus_path):
                self._compute_corpus_statistics(corpus_path)
        
        # Initialize adaptive curvature components if enabled
        if self.use_adaptive_curvature:
            # Replace static curvature with a trainable parameter
            self.static_curvature = self.curvature  # Store the static value
            self.curvature = torch.nn.Parameter(torch.tensor(curvature, device=self.device))
            
            # Create optimizer for curvature
            self.curvature_optimizer = torch.optim.Adam([self.curvature], lr=curvature_lr)
            
            # Store additional parameters
            self.hierarchy_weight = hierarchy_weight
            self.distortion_weight = distortion_weight
            self.optimize_curvature_freq = optimize_curvature_freq
            
            # Track merge pairs for hierarchy information
            self.merge_pairs: List[Tuple[int, int]] = []
            
            # Project embeddings to ensure they lie on the hyperboloid with current curvature
            with torch.no_grad():
                self.embeddings.data = project_to_hyperboloid(self.embeddings.data, self.curvature.item())
        
        # Initialize compression-aware components if enabled
        if self.use_compression_aware:
            self.compression_weight = compression_weight
            self.distance_weight = distance_weight
            self.sample_size = sample_size
            
            # Store corpus sample for compression evaluation
            self.corpus_sample = corpus_sample or []
            
            # Cache for tokenization results
            self.tokenize_cache = {}
        
        logger.info(f"Initialized EnhancedFastHyperbolicTokenizer with features: "
                    f"frequency={use_frequency_aware}, "
                    f"hierarchical={use_hierarchical}, "
                    f"adaptive_curvature={use_adaptive_curvature}, "
                    f"compression={use_compression_aware}")
    
    # --- Frequency-aware methods ---
    
    def _compute_pair_frequencies(self, corpus_path: str) -> None:
        """
        Compute token pair frequencies from corpus.
        
        Args:
            corpus_path: Path to corpus file
        """
        if not self.use_frequency_aware:
            return
            
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
        if not self.use_frequency_aware:
            return 0.0
            
        with torch.no_grad():
            # Get embeddings
            emb_i = self.embeddings[i].unsqueeze(0)
            emb_j = self.embeddings[j].unsqueeze(0)
            
            # Compute midpoint in tangent space (this simulates the merged embedding)
            token_i, token_j = self.vocab[i], self.vocab[j]
            weight_i = len(token_i) / (len(token_i) + len(token_j))
            weight_j = len(token_j) / (len(token_i) + len(token_j))
            
            # Compute the midpoint in tangent space
            v_j_at_i = log_map(emb_i, emb_j, self.get_curvature())
            v_scaled = v_j_at_i * weight_j
            merged_emb = exp_map(emb_i, v_scaled, self.get_curvature())
            
            # Sample a small set of tokens to compare against
            sample_size = min(50, self.current_vocab_size)
            indices = torch.randperm(self.current_vocab_size)[:sample_size]
            
            # Compute distances from merged token to sampled tokens
            distances = []
            for idx in indices:
                if idx != i and idx != j:  # Skip the tokens being merged
                    emb_k = self.embeddings[idx].unsqueeze(0)
                    dist = distance(merged_emb, emb_k, self.get_curvature()).item()
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
    
    def _compute_frequency_score(self, i: int, j: int) -> float:
        """
        Compute frequency score for merging tokens i and j.
        
        Args:
            i: Index of first token
            j: Index of second token
            
        Returns:
            Frequency score (higher is better)
        """
        if not self.use_frequency_aware or not self.pair_frequencies:
            return 0.0
            
        token_i, token_j = self.vocab[i], self.vocab[j]
        
        # Frequency score based on pair occurrence
        pair_freq = self.pair_frequencies.get((token_i, token_j), 0)
        freq_score = np.log1p(pair_freq)  # Log to handle skewed distributions
        
        # Normalize frequency score to 0-1 range
        max_freq = max(self.pair_frequencies.values()) if self.pair_frequencies else 1
        freq_score = freq_score / np.log1p(max_freq) if max_freq > 0 else 0
        
        return freq_score
    
    def get_curvature(self) -> Union[float, torch.Tensor]:
        """
        Get current curvature value, handling both static and dynamic cases.
        
        Returns:
            Current curvature value
        """
        if self.use_adaptive_curvature:
            return self.curvature
        else:
            return self.static_curvature if hasattr(self, 'static_curvature') else self.curvature
    
    # --- Hierarchical merge strategy methods ---
    
    def _compute_corpus_statistics(self, corpus_path: str) -> None:
        """
        Compute corpus statistics for guiding hierarchical merges.
        
        Args:
            corpus_path: Path to corpus file
        """
        if not self.use_hierarchical:
            return
            
        logger.info("Computing corpus statistics for hierarchical merging...")
        word_counter = Counter()
        subword_counter = Counter()
        
        # Process the corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing corpus"):
                # Tokenize into words first
                words = re.findall(r'\b\w+\b', line.lower())
                
                # Count words
                word_counter.update(words)
                
                # Extract potential morphemes (substrings)
                for word in words:
                    # Count character n-grams as potential subwords
                    for n in range(2, min(6, len(word) + 1)):
                        for i in range(len(word) - n + 1):
                            subword = word[i:i+n]
                            subword_counter[subword] += 1
        
        # Store token frequencies
        self.token_frequencies = dict(word_counter)
        
        # Identify common morphemes (frequent subwords)
        subword_threshold = np.percentile(list(subword_counter.values()), 80)
        self.common_morphemes = {
            subword for subword, count in subword_counter.items()
            if count >= subword_threshold
        }
        
        # Identify common words
        word_threshold = np.percentile(list(word_counter.values()), 70)
        self.common_words = {
            word for word, count in word_counter.items()
            if count >= word_threshold
        }
        
        logger.info(f"Identified {len(self.common_morphemes)} common morphemes and "
                   f"{len(self.common_words)} common words")
    
    def _is_potential_morpheme(self, token: str) -> bool:
        """
        Check if a token is a potential morpheme.
        
        Uses a combination of corpus statistics and linguistic rules.
        
        Args:
            token: Token to check
            
        Returns:
            Whether the token is a potential morpheme
        """
        if not self.use_hierarchical:
            return True
            
        # Check if it's in our identified common morphemes
        if token in self.common_morphemes:
            return True
        
        # Check based on morphological rules
        if NLTK_AVAILABLE:
            # Common prefixes in English
            common_prefixes = {'re', 'un', 'in', 'im', 'il', 'ir', 'dis', 'en', 'em', 'non', 'de', 'pre', 'pro', 'mis'}
            # Common suffixes in English
            common_suffixes = {'ing', 'ed', 'er', 'est', 'ly', 'ity', 'ment', 'ness', 'able', 'ible', 'al', 'ial'}
            
            # Check if token is a common prefix or suffix
            if token in common_prefixes or token in common_suffixes:
                return True
            
            # Try to find in WordNet
            if len(token) > 2:  # Minimum length for meaningful morpheme
                # Check if any word with this morpheme exists in WordNet
                for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
                    if wordnet.synsets(token, pos=pos):
                        return True
        
        # Fallback heuristics
        if len(token) >= 2 and len(token) <= 5:
            # If the token appears frequently as a substring in the corpus
            frequency = sum(1 for word in self.common_words if token in word)
            if frequency >= 5:  # Arbitrary threshold
                return True
        
        return False
    
    def _is_valid_word(self, token: str) -> bool:
        """
        Check if a token is a valid word.
        
        Args:
            token: Token to check
            
        Returns:
            Whether the token is a valid word
        """
        if not self.use_hierarchical:
            return True
            
        # Check if it's in our common words
        if token in self.common_words:
            return True
        
        # Check if it's a dictionary word using NLTK
        if NLTK_AVAILABLE:
            # Check if it exists in WordNet
            if wordnet.synsets(token):
                return True
        
        # Fallback: check if it's long enough and has vowels (basic heuristic)
        if len(token) >= 3 and re.search(r'[aeiou]', token):
            return True
            
        return False
    
    def _get_merge_phase_threshold(self) -> float:
        """
        Get appropriate merge threshold for current hierarchical phase.
        
        Returns:
            Merge threshold for current phase
        """
        if not self.use_hierarchical:
            return self.merge_threshold
            
        # Phase-specific thresholds
        if self.current_phase == 1:  # Character-level phase
            return 0.05
        elif self.current_phase == 2:  # Subword-level phase
            return 0.1
        else:  # Word-level phase
            return 0.2
    
    def _filter_by_current_phase(self, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        """
        Filter candidates based on current hierarchical merge phase.
        
        Args:
            candidates: List of merge candidates
            
        Returns:
            Filtered list of candidates
        """
        if not self.use_hierarchical or not candidates:
            return candidates
            
        filtered_candidates = []
        
        # Phase 1: Character-level merges (build basic subwords)
        if self.current_phase == 1:
            for candidate in candidates:
                token_i = self.vocab[candidate.token_i]
                token_j = self.vocab[candidate.token_j]
                
                # Prioritize character-level merges
                if len(token_i) <= 2 and len(token_j) <= 2:
                    # Give higher priority (lower distance)
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance * 0.9,  # Lower = better
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=0.8  # High score for phase 1
                        )
                    )
                else:
                    # Keep but with original distance
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance,
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=0.2
                        )
                    )
        
        # Phase 2: Subword-level merges (build morphemes)
        elif self.current_phase == 2:
            for candidate in candidates:
                token_i = self.vocab[candidate.token_i]
                token_j = self.vocab[candidate.token_j]
                merged = token_i + token_j
                
                # Check if merged token forms a potential morpheme
                if self._is_potential_morpheme(merged):
                    # Give higher priority to morphologically valid merges
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance * 0.8,  # Lower = better
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=0.9
                        )
                    )
                else:
                    # Keep other candidates but with lower priority
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance,
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=0.3
                        )
                    )
        
        # Phase 3: Word-level merges (build compounds and common words)
        else:
            for candidate in candidates:
                token_i = self.vocab[candidate.token_i]
                token_j = self.vocab[candidate.token_j]
                merged = token_i + token_j
                
                # Check if merged token forms a valid word
                if self._is_valid_word(merged):
                    # Give higher priority to valid word merges
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance * 0.7,  # Lower = better
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=1.0
                        )
                    )
                else:
                    # Keep other candidates but with lower priority
                    filtered_candidates.append(
                        EnhancedMergeCandidate(
                            distance=candidate.distance,
                            token_i=candidate.token_i,
                            token_j=candidate.token_j,
                            morphology_score=0.4
                        )
                    )
                    
        return filtered_candidates if filtered_candidates else candidates
    
    # --- Adaptive curvature methods ---
    
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
        if not self.use_adaptive_curvature or not hasattr(self, 'merge_pairs') or not self.merge_pairs:
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
        if not self.use_adaptive_curvature:
            return torch.tensor(0.0, device=self.device)
            
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
        if not self.use_adaptive_curvature:
            return
            
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
        if not self.use_adaptive_curvature:
            return
            
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
        # Track merge pair for hierarchy preservation if using adaptive curvature
        if self.use_adaptive_curvature:
            self.merge_pairs.append((i, j))
        
        # Call parent implementation
        super()._merge_tokens(i, j)
    
    # --- Compression-aware methods ---
    
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
        if not self.use_compression_aware:
            return self.tokenize(text)
            
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
    
    def _compute_compression_score(self, i: int, j: int) -> float:
        """
        Compute compression score for merging tokens i and j.
        
        Args:
            i: Index of first token
            j: Index of second token
            
        Returns:
            Compression score (higher is better)
        """
        if not self.use_compression_aware or not self.corpus_sample:
            return 0.0
            
        # Compute original tokenization length once
        cache_key = "original"
        if cache_key not in self.tokenize_cache:
            self.tokenize_cache[cache_key] = sum(len(self.tokenize(text)) for text in self.corpus_sample)
        original_tokens = self.tokenize_cache[cache_key]
        
        # Simulate merge
        merged_token = self.vocab[i] + self.vocab[j]
        temp_vocab = self.vocab.copy()
        temp_vocab.append(merged_token)
        
        # Compute compression ratio
        merged_tokens = 0
        sample_size = min(len(self.corpus_sample), 10)  # Limit to 10 samples for efficiency
        for text in self.corpus_sample[:sample_size]:
            # Use cached tokenization if available
            cache_key = f"merge_{i}_{j}_{text[:20]}"
            if cache_key not in self.tokenize_cache:
                # Simple tokenization for evaluation
                tokens = self._tokenize_with_vocab(text, temp_vocab)
                self.tokenize_cache[cache_key] = len(tokens)
            
            merged_tokens += self.tokenize_cache[cache_key]
        
        # Scale up based on sample size
        if sample_size < len(self.corpus_sample):
            merged_tokens = merged_tokens * (len(self.corpus_sample) / sample_size)
        
        # Avoid division by zero
        if merged_tokens == 0:
            compression_ratio = 1.0
        else:
            compression_ratio = original_tokens / merged_tokens
        
        # Normalize to 0-1 range (typical compression ratios are 1.0-2.0)
        normalized_score = min(1.0, (compression_ratio - 1.0) / 1.0)
        return max(0.0, normalized_score)
    
    # --- Integrated core methods ---
    
    def _score_candidate(self, candidate: MergeCandidate) -> EnhancedMergeCandidate:
        """
        Compute integrated score for a merge candidate using all enabled features.
        
        Args:
            candidate: Merge candidate to score
            
        Returns:
            Enhanced merge candidate with all scores
        """
        i, j = candidate.token_i, candidate.token_j
        dist = candidate.distance
        
        # Base distance score (lower distance = higher score)
        distance_score = 1.0 / (1.0 + dist)
        
        # Initialize all scores
        frequency_score = 0.0
        semantic_score = 0.0
        compression_score = 0.0
        morphology_score = 0.0
        
        # Compute feature-specific scores if enabled
        if self.use_frequency_aware:
            frequency_score = self._compute_frequency_score(i, j)
            semantic_score = self._compute_semantic_coherence(i, j)
        
        if self.use_compression_aware:
            compression_score = self._compute_compression_score(i, j)
        
        # Morphology score is handled by _filter_by_current_phase
        if self.use_hierarchical:
            token_i, token_j = self.vocab[i], self.vocab[j]
            merged = token_i + token_j
            
            if self.current_phase == 1:  # Character phase
                morphology_score = 0.8 if len(token_i) <= 2 and len(token_j) <= 2 else 0.2
            elif self.current_phase == 2:  # Subword phase
                morphology_score = 0.9 if self._is_potential_morpheme(merged) else 0.3
            else:  # Word phase
                morphology_score = 1.0 if self._is_valid_word(merged) else 0.4
        
        # Combine scores with appropriate weights
        # Default weights if not using frequency-aware
        alpha, beta, gamma = 0.7, 0.0, 0.0
        if self.use_frequency_aware:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            
        # Default compression weight
        compression_weight = 0.0
        if self.use_compression_aware:
            compression_weight = self.compression_weight
            alpha *= (1 - compression_weight)  # Scale down other weights
            beta *= (1 - compression_weight)
            gamma *= (1 - compression_weight)
        
        # Default morphology weight
        morphology_weight = 0.0
        if self.use_hierarchical:
            morphology_weight = 0.3
            # Scale down other weights
            alpha *= (1 - morphology_weight)
            beta *= (1 - morphology_weight)
            gamma *= (1 - morphology_weight)
            if self.use_compression_aware:
                compression_weight *= (1 - morphology_weight)
        
        # Compute combined score
        combined_score = (
            alpha * distance_score +
            beta * frequency_score +
            gamma * semantic_score +
            compression_weight * compression_score +
            morphology_weight * morphology_score
        )
        
        # Create enhanced candidate with all scores
        # We negate the combined score because the heap is min-heap, but we want higher scores to be better
        return EnhancedMergeCandidate(
            distance=dist,
            token_i=i,
            token_j=j,
            frequency_score=frequency_score,
            semantic_score=semantic_score,
            compression_score=compression_score,
            morphology_score=morphology_score,
            combined_score=-combined_score  # Negate for min-heap
        )
    
    def _find_merge_candidates_fast(self) -> List[EnhancedMergeCandidate]:
        """
        Find merge candidates using enhanced scoring based on all enabled features.
        
        Returns:
            List of enhanced merge candidates sorted by combined score
        """
        # Use the parent implementation to get basic candidates
        basic_candidates = super()._find_merge_candidates_fast()
        
        # If no features are enabled, return basic candidates
        if not (self.use_frequency_aware or self.use_hierarchical or 
                self.use_compression_aware or self.use_adaptive_curvature):
            return basic_candidates
        
        # Convert basic candidates to enhanced candidates with combined scoring
        enhanced_candidates = [self._score_candidate(candidate) for candidate in basic_candidates]
        
        # Sort by combined score (already negated for min-heap)
        enhanced_candidates.sort()
        
        return enhanced_candidates
    
    def optimize_merges(self, 
                        steps: int = 10000, 
                        log_every: int = 1000,
                        corpus_sample: Optional[List[str]] = None,
                        adaptive_threshold: bool = True,
                        phase_transition_steps: Optional[Dict[int, int]] = None) -> None:
        """
        Perform iterative merge optimization with all enabled features.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            corpus_sample: Optional corpus sample for compression evaluation
            adaptive_threshold: Whether to adaptively adjust merge threshold
            phase_transition_steps: Dictionary mapping phase numbers to steps at which to transition
                                   Default: {2: 1000, 3: 6000} (phase 1 → 2 at step 1000, phase 2 → 3 at step 6000)
        """
        # Update corpus sample if provided and using compression-aware
        if corpus_sample and self.use_compression_aware:
            self.corpus_sample = corpus_sample
            # Clear cache when corpus changes
            self.tokenize_cache = {}
        
        # Set up phase transitions if using hierarchical
        if self.use_hierarchical:
            if phase_transition_steps is None:
                phase_transition_steps = {2: 1000, 3: 6000}
        
        # Perform optimization
        from tqdm import tqdm
        import random
        
        pbar = tqdm(range(steps), desc="Optimizing merges")
        no_candidate_count = 0
        stats = {}
        
        # Set initial threshold based on hierarchical phase if enabled
        if self.use_hierarchical:
            self.merge_threshold = self._get_merge_phase_threshold()
            logger.info(f"Starting with phase {self.current_phase} threshold: {self.merge_threshold:.4f}")
        
        for step in pbar:
            # Check for phase transitions if using hierarchical
            if self.use_hierarchical and step in phase_transition_steps.values():
                for phase, transition_step in phase_transition_steps.items():
                    if step == transition_step:
                        self.current_phase = phase
                        self.merge_threshold = self._get_merge_phase_threshold()
                        logger.info(f"Transitioning to phase {self.current_phase} with threshold: {self.merge_threshold:.4f}")
                        # Clear caches when changing phases
                        if hasattr(self, 'tokenize_cache'):
                            self.tokenize_cache = {}
            
            # Periodically optimize curvature if enabled
            if self.use_adaptive_curvature and step > 0 and step % self.optimize_curvature_freq == 0:
                # Detach embeddings for optimization to avoid affecting the merge
                self._optimize_curvature(self.embeddings.detach())
                
                # Project embeddings to ensure they lie on the hyperboloid with updated curvature
                self._project_embeddings()
            
            # Compute distance statistics periodically
            if step % log_every == 0 and adaptive_threshold:
                with torch.no_grad():
                    # Sample distances for statistics
                    n = self.current_vocab_size
                    sample_size = min(1000, n * (n - 1) // 2)
                    sample_dists = []
                    
                    for _ in range(sample_size):
                        i, j = random.sample(range(n), 2)
                        d = distance(
                            self.embeddings[i].unsqueeze(0),
                            self.embeddings[j].unsqueeze(0),
                            self.get_curvature()
                        ).item()
                        sample_dists.append(d)
                    
                    if sample_dists:
                        min_dist = min(sample_dists)
                        max_dist = max(sample_dists)
                        mean_dist = np.mean(sample_dists)
                        
                        # Log statistics
                        logger.info(f"\nStep {step}: vocab_size={self.current_vocab_size}")
                        logger.info(f"  Distance stats: min={min_dist:.6f}, "
                                  f"max={max_dist:.6f}, mean={mean_dist:.6f}")
                        logger.info(f"  Merge threshold: {self.merge_threshold:.6f}")
                        
                        # Store stats
                        stats[step] = {
                            "vocab_size": self.current_vocab_size,
                            "min_dist": min_dist,
                            "max_dist": max_dist,
                            "mean_dist": mean_dist,
                            "phase": self.current_phase if self.use_hierarchical else 0
                        }
            
            # Find merge candidates
            candidates = self._find_merge_candidates_fast()
            
            if not candidates:
                no_candidate_count += 1
                
                if no_candidate_count > 5 and adaptive_threshold:
                    # Increase threshold if we can't find candidates
                    old_threshold = self.merge_threshold
                    self.merge_threshold *= 1.5
                    logger.info(f"No candidates found. Increasing threshold from {old_threshold:.6f} to {self.merge_threshold:.6f}")
                    no_candidate_count = 0
                    continue
                elif no_candidate_count > 10:
                    logger.info(f"No more merge candidates found after {step} steps")
                    break
                continue
            else:
                no_candidate_count = 0
            
            # Take the best candidate
            best = candidates[0]
            
            # Convert back to basic MergeCandidate if needed
            if isinstance(best, EnhancedMergeCandidate):
                i, j, dist = best.token_i, best.token_j, best.distance
                combined_score = -best.combined_score  # Un-negate for display
                
                # Log detailed scores
                if step % log_every == 0:
                    logger.info(f"  Best candidate scores: "
                               f"distance={best.distance:.4f}, "
                               f"frequency={best.frequency_score:.4f}, "
                               f"semantic={best.semantic_score:.4f}, "
                               f"compression={best.compression_score:.4f}, "
                               f"morphology={best.morphology_score:.4f}, "
                               f"combined={combined_score:.4f}")
            else:
                i, j, dist = best.token_i, best.token_j, best.distance
                combined_score = 1.0 / (1.0 + dist)  # Simple score for display
            
            # Perform the merge
            self._merge_tokens(i, j)
            
            # Clear part of the cache that might be affected by this merge
            if hasattr(self, 'tokenize_cache') and self.tokenize_cache:
                keys_to_remove = [k for k in self.tokenize_cache if k.startswith("merge_")]
                for k in keys_to_remove:
                    self.tokenize_cache.pop(k, None)
            
            # Update progress
            pbar_postfix = {
                "vocab_size": self.current_vocab_size,
                "score": f"{combined_score:.4f}",
                "threshold": f"{self.merge_threshold:.4f}"
            }
            
            # Add curvature to progress bar if using adaptive curvature
            if self.use_adaptive_curvature:
                pbar_postfix["curvature"] = f"{self.curvature.item():.4f}"
                
            # Add phase to progress bar if using hierarchical
            if self.use_hierarchical:
                pbar_postfix["phase"] = self.current_phase
                
            pbar.set_postfix(pbar_postfix)
            
            # Log progress
            if (step + 1) % log_every == 0:
                token_i, token_j = self.vocab[i], self.vocab[j]
                merged_token = self.vocab[-1]
                logger.info(
                    f"Step {step+1}: merged '{token_i}' + '{token_j}' -> '{merged_token}' "
                    f"(score: {combined_score:.4f}, phase: {self.current_phase if self.use_hierarchical else 0})"
                )
            
            # Adaptive threshold adjustment if enabled
            if adaptive_threshold and step > 0 and step % 1000 == 0:
                # For hierarchical merging, use phase-specific threshold
                if self.use_hierarchical:
                    self.merge_threshold = self._get_merge_phase_threshold() * (1.1 ** (step // 1000))
                else:
                    # Gradual increase
                    self.merge_threshold *= 1.1
        
        # Save final statistics
        if stats:
            # Save stats as a property of the tokenizer
            self.training_stats = stats
            
            # Log final statistics
            logger.info(f"\nCompleted optimization with {len(self.vocab)} tokens")
            logger.info(f"Final merge threshold: {self.merge_threshold:.6f}")
            if self.use_adaptive_curvature:
                logger.info(f"Final curvature: {self.curvature.item():.6f}")
            if self.use_hierarchical:
                logger.info(f"Final phase: {self.current_phase}")
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk, including all enhanced features.
        
        Args:
            path: Directory path to save the tokenizer
        """
        import os
        import json
        
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.vocab, f)
        
        # Save embeddings
        torch.save(self.embeddings, f"{path}/embeddings.pt")
        
        # Save merge history
        with open(f"{path}/merges.json", "w") as f:
            json.dump(self.merge_history, f)
        
        # Save curvature
        if self.use_adaptive_curvature:
            torch.save(self.curvature, f"{path}/curvature.pt")
        
        # Save enhanced configuration
        config = {
            "curvature": self.get_curvature().item() if hasattr(self.get_curvature(), "item") else self.get_curvature(),
            "merge_threshold": self.merge_threshold,
            "max_vocab_size": self.max_vocab_size,
            "use_approximate_search": self.use_approximate_search,
            
            # Feature flags
            "use_frequency_aware": self.use_frequency_aware,
            "use_hierarchical": self.use_hierarchical,
            "use_adaptive_curvature": self.use_adaptive_curvature,
            "use_compression_aware": self.use_compression_aware,
            
            # Feature-specific parameters
            "alpha": getattr(self, "alpha", 0.4),
            "beta": getattr(self, "beta", 0.4),
            "gamma": getattr(self, "gamma", 0.2),
            "language": getattr(self, "language", "english"),
            "hierarchy_weight": getattr(self, "hierarchy_weight", 1.0),
            "distortion_weight": getattr(self, "distortion_weight", 0.1),
            "compression_weight": getattr(self, "compression_weight", 0.7),
            "distance_weight": getattr(self, "distance_weight", 0.3),
            
            # Current state
            "current_phase": getattr(self, "current_phase", 1),
            "current_vocab_size": self.current_vocab_size
        }
        
        with open(f"{path}/enhanced_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save training statistics if available
        if hasattr(self, "training_stats") and self.training_stats:
            with open(f"{path}/training_stats.json", "w") as f:
                # Convert steps (keys) to strings for JSON
                serializable_stats = {str(k): v for k, v in self.training_stats.items()}
                json.dump(serializable_stats, f, indent=2)
        
        # Save feature-specific data
        
        # Frequency data
        if self.use_frequency_aware and hasattr(self, "pair_frequencies") and self.pair_frequencies:
            # Convert tuple keys to strings for JSON
            frequencies_json = {f"{k[0]}|{k[1]}": v for k, v in self.pair_frequencies.items()}
            with open(f"{path}/frequencies.json", "w") as f:
                json.dump(frequencies_json, f)
        
        # Hierarchical data
        if self.use_hierarchical and (hasattr(self, "common_morphemes") or hasattr(self, "common_words")):
            hierarchical_data = {
                "language": getattr(self, "language", "english"),
                "common_morphemes": list(getattr(self, "common_morphemes", set())),
                "common_words": list(getattr(self, "common_words", set()))
            }
            with open(f"{path}/hierarchical_data.json", "w") as f:
                json.dump(hierarchical_data, f)
        
        # Adaptive curvature data
        if self.use_adaptive_curvature and hasattr(self, "merge_pairs"):
            torch.save(self.merge_pairs, f"{path}/merge_pairs.pt")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'EnhancedFastHyperbolicTokenizer':
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
        
        # Load configuration
        try:
            with open(f"{path}/enhanced_config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            # Try legacy config format
            with open(f"{path}/config.json", "r") as f:
                config = json.load(f)
                # Add default values for enhanced features
                config.update({
                    "use_frequency_aware": False,
                    "use_hierarchical": False,
                    "use_adaptive_curvature": False,
                    "use_compression_aware": False
                })
        
        # Create tokenizer with loaded configuration
        tokenizer = cls(
            vocab=vocab,
            embeddings=embeddings,
            curvature=config.get("curvature", 1.0),
            merge_threshold=config.get("merge_threshold", 0.1),
            device=device,
            max_vocab_size=config.get("max_vocab_size", 100000),
            use_approximate_search=config.get("use_approximate_search", True),
            
            # Feature flags
            use_frequency_aware=config.get("use_frequency_aware", False),
            use_hierarchical=config.get("use_hierarchical", False),
            use_adaptive_curvature=config.get("use_adaptive_curvature", False),
            use_compression_aware=config.get("use_compression_aware", False),
            
            # Feature-specific parameters
            alpha=config.get("alpha", 0.4),
            beta=config.get("beta", 0.4),
            gamma=config.get("gamma", 0.2),
            language=config.get("language", "english"),
            hierarchy_weight=config.get("hierarchy_weight", 1.0),
            distortion_weight=config.get("distortion_weight", 0.1),
            compression_weight=config.get("compression_weight", 0.7),
            distance_weight=config.get("distance_weight", 0.3),
        )
        
        # Load merge history
        with open(f"{path}/merges.json", "r") as f:
            tokenizer.merge_history = json.load(f)
        
        # Set current phase and vocab size
        tokenizer.current_phase = config.get("current_phase", 1)
        tokenizer.current_vocab_size = config.get("current_vocab_size", len(tokenizer.vocab))
        
        # Load feature-specific data
        
        # Adaptive curvature data
        if tokenizer.use_adaptive_curvature:
            try:
                # Load curvature tensor
                tokenizer.curvature = torch.load(f"{path}/curvature.pt", map_location=device)
                
                # Load merge pairs
                tokenizer.merge_pairs = torch.load(f"{path}/merge_pairs.pt", map_location=device)
                
                # Re-create optimizer
                tokenizer.curvature_optimizer = torch.optim.Adam([tokenizer.curvature], 
                                                               lr=config.get("curvature_lr", 0.01))
            except FileNotFoundError:
                logger.warning("Could not load adaptive curvature data")
        
        # Frequency data
        if tokenizer.use_frequency_aware:
            try:
                with open(f"{path}/frequencies.json", "r") as f:
                    frequencies_json = json.load(f)
                    
                    # Convert serialized keys back to tuples
                    tokenizer.pair_frequencies = {
                        tuple(k.split("|")): v for k, v in frequencies_json.items()
                    }
            except FileNotFoundError:
                logger.warning("Could not load frequency data")
        
        # Hierarchical data
        if tokenizer.use_hierarchical:
            try:
                with open(f"{path}/hierarchical_data.json", "r") as f:
                    hierarchical_data = json.load(f)
                    tokenizer.language = hierarchical_data.get("language", "english")
                    tokenizer.common_morphemes = set(hierarchical_data.get("common_morphemes", []))
                    tokenizer.common_words = set(hierarchical_data.get("common_words", []))
            except FileNotFoundError:
                logger.warning("Could not load hierarchical data")
        
        # Load training statistics if available
        try:
            with open(f"{path}/training_stats.json", "r") as f:
                stats_json = json.load(f)
                # Convert string keys back to integers
                tokenizer.training_stats = {int(k): v for k, v in stats_json.items()}
        except FileNotFoundError:
            pass  # Training stats are optional
        
        return tokenizer
