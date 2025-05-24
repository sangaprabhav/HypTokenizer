#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate hierarchy distortion of the tokenizer embeddings.

This script measures how well the hyperbolic embeddings preserve
the hierarchical structure from WordNet.
"""

import torch
import random
import numpy as np
import networkx as nx
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
import typer
from pathlib import Path
import sys

# Add parent directory to path to import from embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.lorentz_model import distance


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


def load_wordnet_graph(graph_path: str) -> nx.Graph:
    """
    Load WordNet graph from file.
    
    Args:
        graph_path: Path to WordNet graph pickle
        
    Returns:
        NetworkX graph object
    """
    logger.info(f"Loading WordNet graph from {graph_path}")
    graph = nx.read_gpickle(graph_path)
    logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def create_node_mapping(graph: nx.Graph, vocab: List[str]) -> Dict[str, int]:
    """
    Create a mapping from WordNet node names to vocabulary indices.
    
    Args:
        graph: WordNet graph
        vocab: Tokenizer vocabulary
        
    Returns:
        Mapping from node names to vocabulary indices
    """
    # Create a mapping of WordNet node names to vocabulary tokens
    # This is a simplified mapping - in practice, you would need more sophisticated matching
    mapping = {}
    
    # Extract word from synset name (format: 'word.pos.id')
    node_words = {node: node.split('.')[0] for node in graph.nodes()}
    
    # Create mapping from node to vocab index
    for node, word in node_words.items():
        if word in vocab:
            mapping[node] = vocab.index(word)
    
    logger.info(f"Created mapping for {len(mapping)}/{graph.number_of_nodes()} nodes")
    return mapping


def compute_distortion(
    graph: nx.Graph,
    embeddings: torch.Tensor,
    node_mapping: Dict[str, int],
    num_pairs: int = 10000,
    curvature: float = 1.0,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute distortion between graph distances and embedding distances.
    
    Args:
        graph: WordNet graph
        embeddings: Token embeddings
        node_mapping: Mapping from node names to vocabulary indices
        num_pairs: Number of node pairs to sample
        curvature: Curvature parameter
        device: Device to use
        
    Returns:
        Tuple of (distortion ratios, distortion statistics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    # Get nodes that have a mapping
    valid_nodes = list(node_mapping.keys())
    logger.info(f"Sampling from {len(valid_nodes)} valid nodes")
    
    # Sample pairs of nodes
    node_pairs = []
    for _ in range(num_pairs):
        # Sample two different nodes
        while True:
            a, b = random.sample(valid_nodes, 2)
            # Check if there's a path between them
            try:
                path_length = nx.shortest_path_length(graph, a, b)
                node_pairs.append((a, b, path_length))
                break
            except nx.NetworkXNoPath:
                # No path between a and b, try again
                continue
    
    logger.info(f"Sampled {len(node_pairs)} node pairs")
    
    # Compute distortion
    ratios = []
    
    for a, b, graph_dist in node_pairs:
        # Get embedding indices
        i, j = node_mapping[a], node_mapping[b]
        
        # Compute hyperbolic distance
        embedding_dist = distance(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0),
            c=curvature
        ).item()
        
        # Compute ratio
        ratio = embedding_dist / graph_dist
        ratios.append(ratio)
    
    ratios_array = np.array(ratios)
    
    # Compute statistics
    stats = {
        "mean": float(np.mean(ratios_array)),
        "median": float(np.median(ratios_array)),
        "min": float(np.min(ratios_array)),
        "max": float(np.max(ratios_array)),
        "std": float(np.std(ratios_array)),
        "num_pairs": len(ratios_array)
    }
    
    logger.info(f"Computed distortion statistics: {stats}")
    
    return ratios_array, stats


def evaluate_hierarchy(
    embeddings_path: str,
    vocab_path: str,
    graph_path: str,
    output_path: str,
    num_pairs: int = 10000,
    curvature: float = 1.0,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate hierarchy distortion of embeddings.
    
    Args:
        embeddings_path: Path to embeddings file
        vocab_path: Path to vocabulary file
        graph_path: Path to WordNet graph pickle
        output_path: Path to save results
        num_pairs: Number of node pairs to sample
        curvature: Curvature parameter
        seed: Random seed
        
    Returns:
        Distortion statistics
    """
    # Set random seeds
    set_seeds(seed)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Load embeddings
    embeddings = torch.load(embeddings_path, map_location=device)
    logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    
    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Load WordNet graph
    graph = load_wordnet_graph(graph_path)
    
    # Create node mapping
    node_mapping = create_node_mapping(graph, vocab)
    
    # Compute distortion
    ratios, stats = compute_distortion(
        graph=graph,
        embeddings=embeddings,
        node_mapping=node_mapping,
        num_pairs=num_pairs,
        curvature=curvature,
        device=device
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save distortion ratios
    np.save(output_path, ratios)
    logger.info(f"Saved distortion ratios to {output_path}")
    
    # Save statistics
    stats_path = os.path.splitext(output_path)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved statistics to {stats_path}")
    
    return stats


def main(
    embeddings_path: str = "results/hyperbolic/v50000/embeddings.pt",
    vocab_path: str = "results/hyperbolic/v50000/vocab.json",
    graph_path: str = "data/processed/wordnet_graph.gpk",
    output_path: str = "results/hyperbolic/v50000/hierarchy_distortion.npy",
    num_pairs: int = 10000,
    curvature: float = 1.0,
    seed: int = 42
) -> None:
    """
    Evaluate hierarchy distortion of embeddings.
    """
    evaluate_hierarchy(
        embeddings_path=embeddings_path,
        vocab_path=vocab_path,
        graph_path=graph_path,
        output_path=output_path,
        num_pairs=num_pairs,
        curvature=curvature,
        seed=seed
    )


if __name__ == "__main__":
    typer.run(main)
