#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an enhanced hyperbolic tokenizer with advanced features.

This script trains the EnhancedFastHyperbolicTokenizer which integrates multiple advanced
tokenization techniques:
1. Frequency-aware merging
2. Hierarchical merge strategy
3. Adaptive curvature optimization
4. Compression-aware scoring
"""

import torch
import json
import random
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import typer
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import from tokenizer and embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.enhanced_fast_hyperbolic_merge import EnhancedFastHyperbolicTokenizer
from embedding.lorentz_model import exp_map, log_map, distance, project_to_hyperboloid


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


def sample_corpus(corpus_path: str, sample_size: int = 1000) -> List[str]:
    """
    Sample lines from corpus for compression evaluation.
    
    Args:
        corpus_path: Path to corpus file
        sample_size: Number of lines to sample
        
    Returns:
        List of sampled lines
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i < sample_size and line.strip():
                lines.append(line.strip())
            elif i >= sample_size:
                break
    
    return lines


def train_enhanced_tokenizer(
    vocab_path: str,
    corpus_path: str,
    output_dir: str,
    embedding_dim: int = 50,
    curvature: float = 1.0,
    initial_merge_threshold: float = 0.1,
    learning_rate: float = 1e-3,
    merge_steps: int = 20000,
    log_every: int = 500,
    target_vocab_size: Optional[int] = None,
    seed: int = 42,
    use_frequency_aware: bool = True,
    use_hierarchical: bool = True,
    use_adaptive_curvature: bool = True,
    use_compression_aware: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 100,
    cache_size: int = 10000,
    rebuild_frequency: int = 100,
    phase_transition_steps: Tuple[int, int] = (1000, 6000),
    no_faiss: bool = False
) -> Dict[str, Any]:
    """
    Train an enhanced hyperbolic tokenizer.
    
    Args:
        vocab_path: Path to initial vocabulary file
        corpus_path: Path to corpus file for frequency analysis
        output_dir: Directory to save the tokenizer
        embedding_dim: Dimension of embeddings
        curvature: Initial curvature parameter
        initial_merge_threshold: Initial threshold for merges
        learning_rate: Learning rate for embedding updates
        merge_steps: Maximum number of merge steps
        log_every: Log frequency
        target_vocab_size: Target vocabulary size
        seed: Random seed
        use_frequency_aware: Whether to use frequency-aware merging
        use_hierarchical: Whether to use hierarchical merge strategy
        use_adaptive_curvature: Whether to optimize curvature dynamically
        use_compression_aware: Whether to use compression-aware scoring
        hnsw_m: HNSW M parameter
        hnsw_ef_construction: HNSW ef_construction parameter
        hnsw_ef_search: HNSW ef_search parameter
        cache_size: Cache size for merge candidates
        rebuild_frequency: How often to rebuild HNSW index
        phase_transition_steps: Steps at which to transition between phases in hierarchical mode
        no_faiss: Whether to avoid using FAISS library
        
    Returns:
        Training statistics
    """
    # Set seeds for reproducibility
    set_seeds(seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    vocab = load_vocab(vocab_path)
    logger.info(f"Loaded initial vocabulary with {len(vocab)} tokens")
    
    # Initialize embeddings
    embeddings = initialize_embeddings(vocab, embedding_dim, curvature, device)
    logger.info(f"Initialized embeddings with shape {embeddings.shape}")
    
    # Create embeddings parameter
    embeddings_param = torch.nn.Parameter(embeddings)
    
    # Sample corpus for compression evaluation
    corpus_sample = sample_corpus(corpus_path)
    logger.info(f"Sampled {len(corpus_sample)} lines from corpus for compression evaluation")
    
    # Initialize tokenizer
    tokenizer = EnhancedFastHyperbolicTokenizer(
        vocab=vocab,
        embeddings=embeddings_param,
        curvature=curvature,
        merge_threshold=initial_merge_threshold,
        learning_rate=learning_rate,
        device=device,
        use_frequency_aware=use_frequency_aware,
        use_hierarchical=use_hierarchical,
        use_adaptive_curvature=use_adaptive_curvature,
        use_compression_aware=use_compression_aware,
        corpus_path=corpus_path,
        corpus_sample=corpus_sample,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        cache_size=cache_size,
        rebuild_frequency=rebuild_frequency,
        use_faiss=not no_faiss
    )
    
    logger.info(f"Initialized EnhancedFastHyperbolicTokenizer with the following features:")
    logger.info(f"  - Frequency-aware merging: {use_frequency_aware}")
    logger.info(f"  - Hierarchical merge strategy: {use_hierarchical}")
    logger.info(f"  - Adaptive curvature optimization: {use_adaptive_curvature}")
    logger.info(f"  - Compression-aware scoring: {use_compression_aware}")
    
    # Training statistics
    stats = {
        "vocab_size_history": [],
        "merge_threshold_history": [],
        "curvature_history": [],
        "merge_time_history": [],
        "features": {
            "frequency_aware": use_frequency_aware,
            "hierarchical": use_hierarchical,
            "adaptive_curvature": use_adaptive_curvature,
            "compression_aware": use_compression_aware
        }
    }
    
    # Training parameters
    optimize_params = {
        "steps": merge_steps,
        "log_every": log_every,
        "target_vocab_size": target_vocab_size,
        "adaptive_threshold": True
    }
    
    # Add phase transition steps for hierarchical mode
    if use_hierarchical and phase_transition_steps:
        phase1_to_2, phase2_to_3 = phase_transition_steps
        optimize_params["phase_transition_steps"] = {
            2: phase1_to_2,  # Step to transition from phase 1 to 2
            3: phase2_to_3   # Step to transition from phase 2 to 3
        }
    
    # Train tokenizer with logging callback
    def log_callback(step, tokenizer):
        if step % log_every == 0:
            stats["vocab_size_history"].append(len(tokenizer.vocab))
            stats["merge_threshold_history"].append(tokenizer.merge_threshold)
            stats["curvature_history"].append(tokenizer.curvature)
            
            # Log current status
            logger.info(f"Step {step}: vocab_size={len(tokenizer.vocab)}, "
                       f"threshold={tokenizer.merge_threshold:.4f}, "
                       f"curvature={tokenizer.curvature:.4f}")
            
            # Add feature-specific stats if available
            if hasattr(tokenizer, 'phase') and use_hierarchical:
                logger.info(f"  Current phase: {tokenizer.phase}")
            
            if hasattr(tokenizer, 'compression_stats') and use_compression_aware:
                compression_ratio = tokenizer.compression_stats.get("avg_compression_ratio", 0)
                logger.info(f"  Compression ratio: {compression_ratio:.4f}")
            
            if hasattr(tokenizer, 'pair_frequencies') and use_frequency_aware:
                num_pairs = len(tokenizer.pair_frequencies)
                logger.info(f"  Number of token pairs tracked: {num_pairs}")
    
    # Register callback and optimize
    tokenizer.register_callback(log_callback)
    tokenizer.optimize_merges(**optimize_params)
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(output_dir)
    logger.info(f"Saved tokenizer to {output_dir}")
    
    # Save training stats
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Final stats and summary
    final_stats = {
        "initial_vocab_size": len(vocab),
        "final_vocab_size": len(tokenizer.vocab),
        "embedding_dim": embedding_dim,
        "initial_curvature": curvature,
        "final_curvature": tokenizer.curvature,
        "merge_steps_performed": len(stats["vocab_size_history"]) * log_every,
        "features_used": {
            "frequency_aware": use_frequency_aware,
            "hierarchical": use_hierarchical,
            "adaptive_curvature": use_adaptive_curvature,
            "compression_aware": use_compression_aware
        }
    }
    
    # Add feature-specific final stats
    if hasattr(tokenizer, 'compression_stats') and use_compression_aware:
        final_stats["compression_ratio"] = tokenizer.compression_stats.get("avg_compression_ratio", 0)
    
    if hasattr(tokenizer, 'hierarchy_stats') and use_hierarchical:
        final_stats["hierarchy_preservation"] = tokenizer.hierarchy_stats.get("avg_preservation_score", 0)
    
    # Save final summary
    with open(os.path.join(output_dir, "final_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    
    logger.info(f"Training complete. Final vocabulary size: {len(tokenizer.vocab)}")
    
    return final_stats


def main(
    vocab_path: str = "data/processed/wiki/vocab_initial.txt",
    corpus_path: str = "data/processed/wiki/wiki.txt",
    output_dir: str = "results/hyperbolic/enhanced_tokenizer",
    embedding_dim: int = 100,
    curvature: float = 1.0,
    initial_merge_threshold: float = 0.1,
    learning_rate: float = 1e-3,
    merge_steps: int = 20000,
    log_every: int = 500,
    target_vocab_size: Optional[int] = 50000,
    seed: int = 42,
    use_frequency_aware: bool = True,
    use_hierarchical: bool = True,
    use_adaptive_curvature: bool = True,
    use_compression_aware: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 100,
    cache_size: int = 10000,
    rebuild_frequency: int = 100,
    phase_transition_steps: str = "1000,6000",
    no_faiss: bool = False
) -> None:
    """
    Train an enhanced hyperbolic tokenizer with the given parameters.
    
    Args:
        vocab_path: Path to initial vocabulary file
        corpus_path: Path to corpus file for frequency analysis
        output_dir: Directory to save the tokenizer
        embedding_dim: Dimension of embeddings
        curvature: Initial curvature parameter
        initial_merge_threshold: Initial threshold for merges
        learning_rate: Learning rate for embedding updates
        merge_steps: Maximum number of merge steps
        log_every: Log frequency
        target_vocab_size: Target vocabulary size
        seed: Random seed
        use_frequency_aware: Whether to use frequency-aware merging
        use_hierarchical: Whether to use hierarchical merge strategy
        use_adaptive_curvature: Whether to optimize curvature dynamically
        use_compression_aware: Whether to use compression-aware scoring
        hnsw_m: HNSW M parameter
        hnsw_ef_construction: HNSW ef_construction parameter
        hnsw_ef_search: HNSW ef_search parameter
        cache_size: Cache size for merge candidates
        rebuild_frequency: How often to rebuild HNSW index
        phase_transition_steps: Comma-separated steps at which to transition between phases
        no_faiss: Whether to avoid using FAISS library
    """
    # Parse phase transition steps
    phase_steps = tuple(int(s.strip()) for s in phase_transition_steps.split(","))
    
    train_enhanced_tokenizer(
        vocab_path=vocab_path,
        corpus_path=corpus_path,
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        curvature=curvature,
        initial_merge_threshold=initial_merge_threshold,
        learning_rate=learning_rate,
        merge_steps=merge_steps,
        log_every=log_every,
        target_vocab_size=target_vocab_size,
        seed=seed,
        use_frequency_aware=use_frequency_aware,
        use_hierarchical=use_hierarchical,
        use_adaptive_curvature=use_adaptive_curvature,
        use_compression_aware=use_compression_aware,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        cache_size=cache_size,
        rebuild_frequency=rebuild_frequency,
        phase_transition_steps=phase_steps,
        no_faiss=no_faiss
    )


if __name__ == "__main__":
    typer.run(main)
