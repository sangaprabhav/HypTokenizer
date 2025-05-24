#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build the WordNet graph for hierarchy evaluation.
Creates a graph of hypernym relationships from WordNet.
"""

import os
import networkx as nx
from nltk.corpus import wordnet as wn
import nltk
import pickle
from typing import Optional
import typer
from pathlib import Path


def build_wordnet_graph(output_path: str = "data/processed/wordnet_graph.gpk") -> nx.Graph:
    """
    Build a graph of WordNet hypernym relationships.
    
    Args:
        output_path: Path to save the graph pickle
    
    Returns:
        The NetworkX graph object
    """
    # Download WordNet if not already available
    nltk.download('wordnet', quiet=True)
    
    print("Building WordNet graph...")
    G = nx.Graph()
    
    # Add edges for hypernym relationships
    for syn in wn.all_synsets('n'):
        for hyp in syn.hypernyms():
            G.add_edge(syn.name(), hyp.name())
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the graph using pickle instead of write_gpickle
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved WordNet graph to {output_path}")
    
    return G


def main(output_path: Optional[str] = "data/processed/wordnet_graph.pkl"):
    """
    Build and save the WordNet graph.
    """
    build_wordnet_graph(output_path)


if __name__ == "__main__":
    typer.run(main)
