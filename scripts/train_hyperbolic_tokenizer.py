#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a hyperbolic tokenizer.

This script trains a tokenizer using hyperbolic geometry to guide subword merges.
"""

import torch
import json
import random
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any
import typer
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import from tokenizer and embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from tokenizer.fast_hyperbolic_merge import FastHyperbolicTokenizer
from embedding.lorentz_model import exp_map, log_map, distance, project_to_hyperboloid, batch_distance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(vocab_path: str) -> List[str]:
    """
    Load vocabulary from file.
    
    Args:
        vocab_path: Path to vocabulary file
        
    Returns:
        List of tokens
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_embeddings(
    vocab: List[str], 
    embedding_dim: int, 
    curvature: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Initialize embeddings in hyperbolic space.
    
    Args:
        vocab: List of tokens
        embedding_dim: Dimension of embeddings
        curvature: Curvature parameter
        device: Device to use
        
    Returns:
        Initialized embeddings tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    # Initialize in tangent space (create vectors in Minkowski space but with first coord as 0)
    # We need embedding_dim+1 for Lorentz model (time + space dimensions)
    tangent = torch.zeros((len(vocab), embedding_dim + 1), dtype=torch.float32, device=device)
    
    # Set the spatial components with small random values
    tangent[:, 1:] = torch.randn((len(vocab), embedding_dim), dtype=torch.float32, device=device) * 0.01
    
    # Origin point in Lorentz model
    origin = torch.zeros(embedding_dim + 1, dtype=torch.float32, device=device)
    origin[0] = 1.0  # First component is 1 for the Lorentz origin
    
    # Map to hyperboloid
    embeddings = torch.zeros((len(vocab), embedding_dim + 1), dtype=torch.float32, device=device)
    
    for i, t in enumerate(tangent):
        # Make sure the tangent vector is orthogonal to the base point in Minkowski space
        # For the origin point, the first component of tangent should be zero (which we set above)        
        embeddings[i] = exp_map(origin, t, curvature)
    
    # Ensure points are on the hyperboloid
    embeddings = project_to_hyperboloid(embeddings, curvature)
    
    return embeddings


def train_tokenizer(
    vocab_path: str,
    output_dir: str,
    embedding_dim: int = 50,
    curvature: float = 1.0,
    merge_threshold: float = 0.1,
    learning_rate: float = 1e-3,
    merge_steps: int = 100000,
    log_every: int = 1000,
    target_vocab_size: Optional[int] = None,
    seed: int = 42,
    use_fast_tokenizer: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 100,
    cache_size: int = 10000,
    rebuild_frequency: int = 100,
    no_faiss: bool = False
) -> Dict[str, Any]:
    """
    Train a hyperbolic tokenizer.
    
    Args:
        vocab_path: Path to initial vocabulary file
        output_dir: Directory to save the tokenizer
        embedding_dim: Dimension of embeddings
        curvature: Curvature parameter
        merge_threshold: Initial threshold for merging tokens
        learning_rate: Learning rate for RSGD optimizer
        merge_steps: Maximum number of merge steps
        log_every: How often to log progress
        target_vocab_size: Target vocabulary size (if None, run for merge_steps)
        seed: Random seed
        
    Returns:
        Dictionary with training statistics
    """
    # Set random seeds
    set_seeds(seed)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    vocab = load_vocab(vocab_path)
    logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Initialize embeddings
    embeddings = initialize_embeddings(vocab, embedding_dim, curvature, device)
    logger.info(f"Initialized embeddings with shape {embeddings.shape}")
    
    # Initialize tokenizer
    if use_fast_tokenizer:
        tokenizer = FastHyperbolicTokenizer(
            vocab=vocab,
            embeddings=torch.nn.Parameter(embeddings),
            curvature=curvature,
            merge_threshold=merge_threshold,
            lr=learning_rate,
            device=device,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            cache_size=cache_size,
            rebuild_frequency=rebuild_frequency,
            use_approximate_search=not no_faiss
        )
        if no_faiss:
            logger.info("Using FastHyperbolicTokenizer with batch optimization (FAISS disabled)")
        else:
            logger.info("Using FastHyperbolicTokenizer with HNSW for efficient nearest neighbor search")
    else:
        tokenizer = HyperbolicTokenizer(
            vocab=vocab,
            embeddings=torch.nn.Parameter(embeddings),
            curvature=curvature,
            merge_threshold=merge_threshold,
            lr=learning_rate,
            device=device
        )
        logger.info("Using standard HyperbolicTokenizer")
    logger.info("Created hyperbolic tokenizer")
    
    # Define optimization callback for logging
    stats = {"vocab_size": [], "distortion": [], "step": []}
    
    def log_callback(step: int, tokenizer: HyperbolicTokenizer) -> None:
        if step % log_every == 0:
            # Record statistics
            stats["vocab_size"].append(len(tokenizer.vocab))
            stats["step"].append(step)
            
            # Compute average distortion (sample a subset of token pairs)
            n = min(1000, len(tokenizer.vocab))
            indices = torch.randperm(len(tokenizer.vocab))[:n]
            emb_sample = tokenizer.embeddings[indices]
            
            # Compute pairwise distances
            dists = torch.zeros((n, n), device=device)
            for i in range(n):
                for j in range(i+1, n):
                    dists[i, j] = distance(
                        emb_sample[i].unsqueeze(0),
                        emb_sample[j].unsqueeze(0),
                        tokenizer.curvature
                    ).item()
                    dists[j, i] = dists[i, j]
            
            avg_dist = dists.sum() / (n * (n - 1))
            stats["distortion"].append(avg_dist.item())
            
            logger.info(f"Step {step}: vocab_size={len(tokenizer.vocab)}, avg_distortion={avg_dist:.4f}")
    
    # Optimize merges
    logger.info(f"Starting merge optimization for {merge_steps} steps")
    
    if use_fast_tokenizer:
        # When using FastHyperbolicTokenizer, we can use its built-in optimize_merges method
        tokenizer.optimize_merges(steps=merge_steps, log_every=log_every)
    else:
        # For the standard tokenizer, create a custom optimize_merges method with callback and target vocab size
        def optimize_with_callback():
            pbar = tqdm(range(merge_steps), desc="Optimizing merges")
            
            for step in pbar:
                # Log callback
                log_callback(step, tokenizer)
                
                # Check if target vocab size reached
                if target_vocab_size is not None and len(tokenizer.vocab) >= target_vocab_size:
                    logger.info(f"Reached target vocabulary size {target_vocab_size}")
                    break
                
                # Find merge candidates
                candidates = tokenizer._find_merge_candidates()
                
                if not candidates:
                    logger.info(f"No more merge candidates found after {step} steps")
                    break
                
                # Sort candidates by distance (ascending)
                candidates.sort(key=lambda x: x[2])
                
                # Process top-k candidates in batch (faster on GPU)
                batch_size = min(64, len(candidates))
                top_candidates = candidates[:batch_size]
                
                # Evaluate distances in parallel
                pairs_i = [c[0] for c in top_candidates]
                pairs_j = [c[1] for c in top_candidates]
                
                # Select the best candidate
                best_idx = 0  # Default to the first candidate
                best_dist = top_candidates[0][2]
                
                # Perform the merge
                i, j, dist = top_candidates[best_idx]
                tokenizer._merge_tokens(i, j)
                
                # Update progress bar
                pbar.set_postfix({
                    "vocab_size": len(tokenizer.vocab),
                    "best_dist": dist,
                    "threshold": tokenizer.merge_threshold
                })
                
                # Adaptive threshold adjustment (optional)
                if step > 0 and step % 1000 == 0:
                    tokenizer.merge_threshold *= 1.05  # Gradually increase threshold
        
        # Run optimization
        optimize_with_callback()
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(output_dir)
    logger.info(f"Saved tokenizer to {output_dir}")
    
    # Save training stats
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f)
    
    return stats


def main(
    vocab_path: str = "data/processed/wiki/vocab_initial.txt",
    output_dir: str = "results/hyperbolic/v50000",
    embedding_dim: int = 5,
    curvature: float = 1.0,
    merge_threshold: float = 0.1,
    learning_rate: float = 1e-3,
    merge_steps: int = 100,
    log_every: int = 10,
    target_vocab_size: Optional[int] = 500,
    seed: int = 42,
    use_fast_tokenizer: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 100,
    cache_size: int = 10000,
    rebuild_frequency: int = 100,
    no_faiss: bool = False
) -> None:
    """
    Train a hyperbolic tokenizer with the given parameters.
    """
    train_tokenizer(
        vocab_path=vocab_path,
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        curvature=curvature,
        merge_threshold=merge_threshold,
        learning_rate=learning_rate,
        merge_steps=merge_steps,
        log_every=log_every,
        target_vocab_size=target_vocab_size,
        seed=seed,
        use_fast_tokenizer=use_fast_tokenizer,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        cache_size=cache_size,
        rebuild_frequency=rebuild_frequency,
        no_faiss=no_faiss
    )


if __name__ == "__main__":
    typer.run(main)
