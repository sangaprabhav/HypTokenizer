#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast Hyperbolic Tokenizer implementation with HNSW.

This module implements a highly optimized version of the hyperbolic tokenizer
using HNSW (Hierarchical Navigable Small World) for fast approximate nearest
neighbor search in hyperbolic space, designed to scale efficiently to large
vocabularies (50k+ tokens).
"""

import torch
import numpy as np
import heapq
import time
import logging
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass
import warnings
import os
import sys

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import distance, batch_distance, exp_map, log_map, project_to_hyperboloid

# Optional imports for fast nearest neighbor search
try:
    import faiss
    # Test if FAISS works on this platform
    try:
        # Simple test to verify FAISS functionality
        d = 10  # Dimension
        index = faiss.IndexFlatL2(d)
        xb = np.random.random((5, d)).astype('float32')
        index.add(xb)
        FAISS_AVAILABLE = True
    except Exception as e:
        FAISS_AVAILABLE = False
        warnings.warn(f"FAISS installation found but failed to initialize: {e}. "  
                     f"Falling back to brute force search.")
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Falling back to brute force search for large vocabularies.")

logger = logging.getLogger(__name__)


@dataclass
class MergeCandidate:
    """Efficient merge candidate storage."""
    distance: float
    token_i: int
    token_j: int
    
    def __lt__(self, other):
        return self.distance < other.distance


class AdaptiveMergeCache:
    """Adaptive caching for merge candidates with usage tracking."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the adaptive merge cache.
        
        Args:
            max_size: Maximum number of candidates to store
        """
        self.max_size = max_size
        self.candidates: List[MergeCandidate] = []
        self.hit_count: Dict[Tuple[int, int], int] = {}
        self.miss_count: int = 0
        
    def add_batch(self, new_candidates: List[MergeCandidate]) -> None:
        """
        Add new candidates with LRU eviction.
        
        Args:
            new_candidates: List of new candidates to add
        """
        # Track usage
        for c in new_candidates:
            key = (c.token_i, c.token_j)
            self.hit_count[key] = self.hit_count.get(key, 0)
        
        # Merge with existing, keeping best
        all_candidates = self.candidates + new_candidates
        all_candidates.sort()
        
        # Keep top candidates with usage weighting
        self.candidates = all_candidates[:self.max_size]
    
    def get_best(self, n: int = 1) -> List[MergeCandidate]:
        """
        Get best candidates and update usage stats.
        
        Args:
            n: Number of candidates to return
            
        Returns:
            List of best candidates
        """
        if not self.candidates:
            self.miss_count += 1
            return []
        
        best = self.candidates[:n]
        for c in best:
            key = (c.token_i, c.token_j)
            self.hit_count[key] = self.hit_count.get(key, 0) + 1
        
        # Remove used candidates
        self.candidates = self.candidates[n:]
        return best
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.candidates),
            "max_size": self.max_size,
            "hit_count": sum(self.hit_count.values()),
            "miss_count": self.miss_count,
            "hit_ratio": sum(self.hit_count.values()) / (sum(self.hit_count.values()) + self.miss_count + 1e-10)
        }


class FastHyperbolicTokenizer(HyperbolicTokenizer):
    """Optimized hyperbolic tokenizer with HNSW index for fast nearest neighbor search."""
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        curvature: float = 1.0,
        merge_threshold: float = 0.1,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True,
        cache_size: int = 10000,
        rebuild_frequency: int = 100,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100
    ):
        """
        Initialize the fast hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            curvature: Curvature parameter of the hyperbolic space
            merge_threshold: Initial threshold for merging tokens
            lr: Learning rate for the RSGD optimizer
            device: Device to use for computation
            max_vocab_size: Maximum vocabulary size to pre-allocate
            use_approximate_search: Whether to use approximate nearest neighbor search
            cache_size: Size of the merge candidates cache
            rebuild_frequency: How often to rebuild the index (in merge steps)
            hnsw_m: HNSW M parameter (number of connections per layer)
            hnsw_ef_construction: HNSW efConstruction parameter (higher = more accurate index)
            hnsw_ef_search: HNSW efSearch parameter (higher = more accurate search)
        """
        # Initialize base class
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
        
        # Initialize additional attributes
        self.index = None
        self.index_outdated = True
        self.cache = AdaptiveMergeCache(max_size=cache_size)
        self.rebuild_frequency = rebuild_frequency
        self.merges_since_rebuild = 0
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        
    def _build_faiss_index(self) -> None:
        """Build FAISS HNSW index for fast nearest neighbor search."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, falling back to brute force search")
            self.use_approximate_search = False
            return
            
        try:
            with torch.no_grad():
                logger.info(f"Building HNSW index for {self.current_vocab_size} tokens...")
                start_time = time.time()
                
                # Convert to Klein model for FAISS (Euclidean space)
                klein_embeddings = self.embeddings[:self.current_vocab_size, 1:] / (
                    self.embeddings[:self.current_vocab_size, 0:1] + 1e-8
                )
                klein_np = klein_embeddings.detach().cpu().numpy().astype('float32')
                
                # Build HNSW index
                dim = klein_np.shape[1]
                self.index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
                self.index.hnsw.efConstruction = self.hnsw_ef_construction
                self.index.hnsw.efSearch = self.hnsw_ef_search
                
                # Use GPU if available
                gpu_index = False
                if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        gpu_res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                        gpu_index = True
                        logger.info("Using GPU-accelerated FAISS index")
                    except Exception as e:
                        logger.warning(f"Failed to use GPU for FAISS: {e}")
                
                # Add all points
                self.index.add(klein_np)
                self.index_outdated = False
                self.merges_since_rebuild = 0
                
                elapsed = time.time() - start_time
                logger.info(f"Built HNSW index in {elapsed:.2f}s on {'GPU' if gpu_index else 'CPU'}")
        except Exception as e:
            logger.warning(f"Failed to build FAISS index: {e}. Falling back to brute force search.")
            self.use_approximate_search = False
            self.index = None
    
    def _find_merge_candidates(self) -> List[Tuple[int, int, float]]:
        """
        Find potential merge candidates based on hyperbolic distance.
        
        Returns:
            List of (i, j, distance) tuples for merge candidates
        """
        # Convert to proper format for base class compatibility
        candidates = self._find_merge_candidates_fast()
        return [(c.token_i, c.token_j, c.distance) for c in candidates]
    
    def _find_merge_candidates_fast(self) -> List[MergeCandidate]:
        """
        Ultra-fast merge candidate finding using HNSW.
        
        Returns:
            List of MergeCandidate objects
        """
        n = self.current_vocab_size
        
        # Check if we can use cached candidates
        cached_candidates = self.cache.get_best(100)
        if cached_candidates:
            return cached_candidates
        
        # Rebuild index if needed
        if (self.index_outdated or self.index is None) and FAISS_AVAILABLE and self.use_approximate_search:
            self._build_faiss_index()
        
        candidates: List[MergeCandidate] = []
        
        # For very large vocabularies, use HNSW approximate search
        if self.use_approximate_search and FAISS_AVAILABLE and n > 10000 and self.index is not None:
            batch_size = min(1000, n)  # Process in batches
            k_neighbors = min(50, n // 10)  # Adaptive neighbor count
            
            with torch.no_grad():
                # Sample tokens to check (stratified sampling for better coverage)
                if n > 5000:
                    # Sample more from recent tokens (likely to merge)
                    recent_weight = 0.7
                    old_size = int(batch_size * (1 - recent_weight))
                    recent_size = batch_size - old_size
                    
                    old_indices = np.random.choice(n // 2, old_size, replace=False)
                    recent_indices = np.random.choice(
                        range(max(n // 2, 1), n), recent_size, replace=False
                    )
                    sample_indices = np.concatenate([old_indices, recent_indices])
                else:
                    sample_indices = np.arange(n)
                
                # Convert sampled embeddings to Klein model
                sample_embeddings = self.embeddings[sample_indices]
                klein_queries = sample_embeddings[:, 1:] / (
                    sample_embeddings[:, 0:1] + 1e-8
                )
                klein_queries_np = klein_queries.detach().cpu().numpy().astype('float32')
                
                # Batch nearest neighbor search
                distances_faiss, indices_faiss = self.index.search(
                    klein_queries_np, k_neighbors
                )
                
                # Convert FAISS distances back to hyperbolic distances
                # and filter by threshold
                candidate_set: Set[Tuple[int, int]] = set()
                
                for i, sample_idx in enumerate(sample_indices):
                    sample_emb = self.embeddings[sample_idx:sample_idx+1]
                    
                    for j in range(k_neighbors):
                        neighbor_idx = indices_faiss[i, j]
                        
                        # Skip invalid indices
                        if neighbor_idx >= n or neighbor_idx <= sample_idx:
                            continue
                        
                        # Compute exact hyperbolic distance for candidates
                        neighbor_emb = self.embeddings[neighbor_idx:neighbor_idx+1]
                        hyp_dist = distance(
                            sample_emb, neighbor_emb, self.curvature
                        ).item()
                        
                        if hyp_dist < self.merge_threshold:
                            pair = (min(sample_idx, neighbor_idx), 
                                   max(sample_idx, neighbor_idx))
                            if pair not in candidate_set:
                                candidate_set.add(pair)
                                candidates.append(
                                    MergeCandidate(hyp_dist, pair[0], pair[1])
                                )
        
        # For medium-sized vocabularies, use batch computation
        elif n > 100:
            with torch.no_grad():
                # Only use the active part of the pre-allocated tensor
                active_embeddings = self.embeddings[:n]
                
                # Compute all pairwise distances using batch_distance
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
                    candidates.append(MergeCandidate(all_dists[i, j].item(), i, j))
        
        # For small vocabularies, use original implementation
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(
                        self.embeddings[i].unsqueeze(0),
                        self.embeddings[j].unsqueeze(0),
                        self.curvature
                    ).item()
                    
                    if dist < self.merge_threshold:
                        candidates.append(MergeCandidate(dist, i, j))
        
        # Sort candidates by distance
        candidates.sort()
        
        # Update cache with new candidates
        self.cache.add_batch(candidates)
        
        return candidates
    
    def _merge_tokens(self, i: int, j: int) -> None:
        """
        Merge two tokens and update the vocabulary and embeddings.
        
        Args:
            i: Index of first token
            j: Index of second token
        """
        # Call the parent implementation
        super()._merge_tokens(i, j)
        
        # Mark index as outdated
        self.merges_since_rebuild += 1
        if self.merges_since_rebuild >= self.rebuild_frequency:
            self.index_outdated = True
    
    def _evaluate_merge_quality_batch(self, candidates: List[MergeCandidate], 
                                     text_sample: List[str]) -> List[float]:
        """
        Evaluate multiple merge candidates in parallel.
        
        Args:
            candidates: List of candidates to evaluate
            text_sample: Sample of text to evaluate on
            
        Returns:
            List of quality scores for each candidate
        """
        with torch.no_grad():
            scores = []
            
            # Simulate merges in parallel
            for candidate in candidates[:100]:  # Top 100
                # Quick quality estimation based on:
                # 1. Frequency of pair in text
                # 2. Semantic coherence
                # 3. Length balance
                
                token_i = self.vocab[candidate.token_i]
                token_j = self.vocab[candidate.token_j]
                merged = token_i + token_j
                
                # Simple frequency-based score
                freq_score = sum(1 for text in text_sample 
                               if token_i + token_j in text)
                
                # Length balance score
                len_score = 1.0 / (1.0 + abs(len(token_i) - len(token_j)))
                
                # Combined score
                score = freq_score * len_score / (candidate.distance + 1e-6)
                scores.append(score)
        
        return scores
    
    def optimize_merges(self, steps: int = 10000, log_every: int = 1000,
                      text_sample: Optional[List[str]] = None) -> None:
        """
        Optimized merge process with batching.
        
        Args:
            steps: Maximum number of merge steps to perform
            log_every: How often to log progress
            text_sample: Optional sample of text for quality evaluation
        """
        from tqdm import tqdm
        
        pbar = tqdm(range(steps), desc="Optimizing merges")
        
        for step in pbar:
            start_time = time.time()
            
            # Get merge candidates
            candidates = self._find_merge_candidates_fast()
            
            if not candidates:
                logger.info(f"No more merge candidates found after {step} steps")
                break
            
            # Take the best candidate
            best = candidates[0]
            
            # Perform the merge
            self._merge_tokens(best.token_i, best.token_j)
            
            # Update progress
            elapsed = time.time() - start_time
            
            # Get cache stats
            cache_stats = self.cache.get_stats()
            
            pbar.set_postfix({
                "vocab_size": self.current_vocab_size,
                "best_dist": best.distance,
                "threshold": self.merge_threshold,
                "time": f"{elapsed:.2f}s",
                "hit_ratio": f"{cache_stats['hit_ratio']:.2f}"
            })
            
            # Log progress
            if (step + 1) % log_every == 0:
                logger.info(
                    f"Step {step+1}: merged '{self.vocab[best.token_i]}' + "
                    f"'{self.vocab[best.token_j]}' -> '{self.vocab[-1]}' "
                    f"(dist: {best.distance:.4f})"
                )
            
            # Adaptive threshold adjustment
            if step > 0 and step % 1000 == 0:
                self.merge_threshold *= 1.02  # Slower increase
