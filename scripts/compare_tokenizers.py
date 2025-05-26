#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare and evaluate different tokenizers.

This script compares various tokenizers including baseline methods (BPE, WordPiece, Unigram),
standard hyperbolic tokenizer, fast hyperbolic tokenizer, and the enhanced hyperbolic tokenizer
with all advanced features. It generates metrics and visualizations for comparison.
"""

import os
import json
import time
import logging
import numpy as np
import torch
import typer
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer as HFTokenizer
import pandas as pd
import sys

# Add parent directory to path to import from tokenizer package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from tokenizer.fast_hyperbolic_merge import FastHyperbolicTokenizer
from tokenizer.enhanced_fast_hyperbolic_merge import EnhancedFastHyperbolicTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str, sample_size: Optional[int] = None) -> List[str]:
    """
    Load corpus from file.
    
    Args:
        corpus_path: Path to corpus file
        sample_size: Number of lines to sample (None for all)
        
    Returns:
        List of text examples
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if sample_size is not None:
        lines = lines[:sample_size]
    
    return lines


def load_huggingface_tokenizer(model_path: str) -> HFTokenizer:
    """
    Load a Hugging Face tokenizer.
    
    Args:
        model_path: Path to tokenizer model
        
    Returns:
        Loaded tokenizer
    """
    return HFTokenizer.from_file(os.path.join(model_path, "tokenizer.json"))


def load_hyperbolic_tokenizer(model_path: str, tokenizer_type: str = "fast") -> Any:
    """
    Load a hyperbolic tokenizer.
    
    Args:
        model_path: Path to tokenizer model
        tokenizer_type: Type of hyperbolic tokenizer ("standard", "fast", or "enhanced")
        
    Returns:
        Loaded tokenizer
    """
    if tokenizer_type == "standard":
        return HyperbolicTokenizer.load(model_path)
    elif tokenizer_type == "fast":
        return FastHyperbolicTokenizer.load(model_path)
    elif tokenizer_type == "enhanced":
        return EnhancedFastHyperbolicTokenizer.load(model_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def benchmark_huggingface_tokenizer(
    tokenizer: HFTokenizer,
    corpus: List[str],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark a Hugging Face tokenizer.
    
    Args:
        tokenizer: Tokenizer to benchmark
        corpus: Corpus to tokenize
        num_runs: Number of runs for timing
        
    Returns:
        Benchmark results
    """
    # Measure tokenization time
    total_time = 0
    total_tokens = 0
    avg_token_length = 0
    
    for _ in range(num_runs):
        start_time = time.time()
        for text in corpus:
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            total_tokens += len(tokens)
            for token in tokens:
                avg_token_length += len(token)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    # Average metrics
    avg_time = total_time / num_runs
    tokens_per_second = total_tokens / avg_time
    avg_tokens_per_text = total_tokens / len(corpus)
    avg_token_length = avg_token_length / total_tokens
    
    # Get vocab size
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    return {
        "vocab_size": vocab_size,
        "avg_tokenization_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "avg_tokens_per_text": avg_tokens_per_text,
        "avg_token_length": avg_token_length,
    }


def benchmark_hyperbolic_tokenizer(
    tokenizer: Any,
    corpus: List[str],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark a hyperbolic tokenizer.
    
    Args:
        tokenizer: Tokenizer to benchmark
        corpus: Corpus to tokenize
        num_runs: Number of runs for timing
        
    Returns:
        Benchmark results
    """
    # Get tokenizer type
    if isinstance(tokenizer, HyperbolicTokenizer):
        tokenizer_type = "standard"
    elif isinstance(tokenizer, FastHyperbolicTokenizer):
        tokenizer_type = "fast"
    elif isinstance(tokenizer, EnhancedFastHyperbolicTokenizer):
        tokenizer_type = "enhanced"
    else:
        tokenizer_type = "unknown"
    
    # Measure tokenization time
    total_time = 0
    total_tokens = 0
    avg_token_length = 0
    
    for _ in range(num_runs):
        start_time = time.time()
        for text in corpus:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)
            for token in tokens:
                avg_token_length += len(token)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    # Average metrics
    avg_time = total_time / num_runs
    tokens_per_second = total_tokens / avg_time
    avg_tokens_per_text = total_tokens / len(corpus)
    avg_token_length = avg_token_length / total_tokens
    
    # Get vocab size
    vocab_size = len(tokenizer.vocab)
    
    # Get additional metrics specific to enhanced tokenizer
    additional_metrics = {}
    if tokenizer_type == "enhanced":
        # Collect feature-specific metrics if available
        if hasattr(tokenizer, "compression_stats"):
            additional_metrics["compression_ratio"] = tokenizer.compression_stats.get("avg_compression_ratio", None)
        
        if hasattr(tokenizer, "hierarchy_stats"):
            additional_metrics["hierarchy_preservation"] = tokenizer.hierarchy_stats.get("avg_preservation_score", None)
        
        if hasattr(tokenizer, "frequency_stats"):
            additional_metrics["frequency_correlation"] = tokenizer.frequency_stats.get("correlation_score", None)
        
        if hasattr(tokenizer, "curvature_history"):
            additional_metrics["final_curvature"] = tokenizer.curvature
            additional_metrics["curvature_changes"] = len(tokenizer.curvature_history)
    
    return {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "avg_tokenization_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "avg_tokens_per_text": avg_tokens_per_text,
        "avg_token_length": avg_token_length,
        **additional_metrics
    }


def evaluate_linguistic_quality(
    tokenizer: Any,
    corpus: List[str],
    is_hyperbolic: bool = False
) -> Dict[str, float]:
    """
    Evaluate linguistic quality of tokenization.
    
    Args:
        tokenizer: Tokenizer to evaluate
        corpus: Corpus to tokenize
        is_hyperbolic: Whether tokenizer is a hyperbolic tokenizer
        
    Returns:
        Linguistic quality metrics
    """
    import re
    
    # Count metrics
    total_tokens = 0
    morpheme_boundary_count = 0
    word_boundary_count = 0
    subword_count = 0
    
    # Regex patterns for linguistic evaluation
    word_boundary_pattern = re.compile(r'[^\w]')
    morpheme_patterns = [
        re.compile(r'(ion|tion|ation|ment|ance|ence|ly|ish|less|ful|ness|ing|ed|er|est|pre|un|re|de|dis)$')
    ]
    
    for text in corpus:
        # Tokenize
        if is_hyperbolic:
            tokens = tokenizer.tokenize(text)
        else:
            tokens = tokenizer.encode(text).tokens
        
        total_tokens += len(tokens)
        
        # Analyze tokens
        for i, token in enumerate(tokens):
            # Check if token contains word boundary
            if word_boundary_pattern.search(token):
                word_boundary_count += 1
            
            # Check if token is a potential morpheme
            for pattern in morpheme_patterns:
                if pattern.search(token):
                    morpheme_boundary_count += 1
                    break
            
            # Check if token is a subword (not a complete word)
            if (i > 0 and not word_boundary_pattern.search(tokens[i-1][-1] + token[0])) or \
               (i < len(tokens) - 1 and not word_boundary_pattern.search(token[-1] + tokens[i+1][0])):
                subword_count += 1
    
    # Calculate metrics
    word_boundary_ratio = word_boundary_count / total_tokens
    morpheme_ratio = morpheme_boundary_count / total_tokens
    subword_ratio = subword_count / total_tokens
    
    return {
        "word_boundary_ratio": word_boundary_ratio,
        "morpheme_ratio": morpheme_ratio,
        "subword_ratio": subword_ratio,
    }


def evaluate_compression_efficiency(
    tokenizer: Any,
    corpus: List[str],
    is_hyperbolic: bool = False
) -> Dict[str, float]:
    """
    Evaluate compression efficiency of tokenization.
    
    Args:
        tokenizer: Tokenizer to evaluate
        corpus: Corpus to tokenize
        is_hyperbolic: Whether tokenizer is a hyperbolic tokenizer
        
    Returns:
        Compression efficiency metrics
    """
    total_chars = 0
    total_tokens = 0
    
    for text in corpus:
        total_chars += len(text)
        
        # Tokenize
        if is_hyperbolic:
            tokens = tokenizer.tokenize(text)
        else:
            tokens = tokenizer.encode(text).tokens
        
        total_tokens += len(tokens)
    
    # Calculate compression metrics
    chars_per_token = total_chars / total_tokens
    compression_ratio = total_chars / (total_tokens * 2)  # Assuming 2 bytes per token ID on average
    
    return {
        "chars_per_token": chars_per_token,
        "compression_ratio": compression_ratio,
    }


def compare_tokenizers(
    corpus_path: str,
    baseline_dir: str,
    hyperbolic_dirs: Dict[str, str],
    output_dir: str,
    sample_size: int = 1000,
    num_runs: int = 3
) -> None:
    """
    Compare different tokenizers and generate evaluation metrics.
    
    Args:
        corpus_path: Path to corpus file
        baseline_dir: Directory containing baseline tokenizers
        hyperbolic_dirs: Dictionary mapping hyperbolic tokenizer types to directories
        output_dir: Directory to save comparison results
        sample_size: Number of examples to use for evaluation
        num_runs: Number of runs for timing benchmarks
    """
    # Load corpus
    logger.info(f"Loading corpus from {corpus_path} (sample size: {sample_size})")
    corpus = load_corpus(corpus_path, sample_size)
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store all results
    all_results = {"baseline": {}, "hyperbolic": {}}
    
    # Benchmark baseline tokenizers
    logger.info("Benchmarking baseline tokenizers")
    baseline_tokenizer_dirs = [d for d in os.listdir(baseline_dir) if os.path.isdir(os.path.join(baseline_dir, d))]
    
    for tokenizer_dir in baseline_tokenizer_dirs:
        tokenizer_path = os.path.join(baseline_dir, tokenizer_dir)
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        
        if os.path.exists(tokenizer_file):
            try:
                # Load tokenizer
                tokenizer = load_huggingface_tokenizer(tokenizer_path)
                
                # Get tokenizer type and vocab size from directory name
                parts = tokenizer_dir.split("_")
                if len(parts) > 1:
                    tokenizer_type = parts[0]
                    vocab_size = parts[1].replace("v", "")
                else:
                    tokenizer_type = tokenizer_dir
                    vocab_size = "unknown"
                
                logger.info(f"Benchmarking {tokenizer_type} tokenizer (vocab size: {vocab_size})")
                
                # Benchmark tokenizer
                benchmark_results = benchmark_huggingface_tokenizer(tokenizer, corpus, num_runs)
                
                # Evaluate linguistic quality
                linguistic_results = evaluate_linguistic_quality(tokenizer, corpus, is_hyperbolic=False)
                
                # Evaluate compression efficiency
                compression_results = evaluate_compression_efficiency(tokenizer, corpus, is_hyperbolic=False)
                
                # Combine results
                all_results["baseline"][tokenizer_dir] = {
                    "tokenizer_type": tokenizer_type,
                    "vocab_size": vocab_size,
                    **benchmark_results,
                    **linguistic_results,
                    **compression_results
                }
            except Exception as e:
                logger.error(f"Error benchmarking {tokenizer_dir}: {e}")
    
    # Benchmark hyperbolic tokenizers
    logger.info("Benchmarking hyperbolic tokenizers")
    for tokenizer_type, tokenizer_dir in hyperbolic_dirs.items():
        try:
            # Load tokenizer
            if tokenizer_type == "standard":
                tokenizer = load_hyperbolic_tokenizer(tokenizer_dir, "standard")
            elif tokenizer_type == "fast":
                tokenizer = load_hyperbolic_tokenizer(tokenizer_dir, "fast")
            elif tokenizer_type == "enhanced":
                tokenizer = load_hyperbolic_tokenizer(tokenizer_dir, "enhanced")
            else:
                logger.error(f"Unknown hyperbolic tokenizer type: {tokenizer_type}")
                continue
            
            logger.info(f"Benchmarking {tokenizer_type} hyperbolic tokenizer")
            
            # Benchmark tokenizer
            benchmark_results = benchmark_hyperbolic_tokenizer(tokenizer, corpus, num_runs)
            
            # Evaluate linguistic quality
            linguistic_results = evaluate_linguistic_quality(tokenizer, corpus, is_hyperbolic=True)
            
            # Evaluate compression efficiency
            compression_results = evaluate_compression_efficiency(tokenizer, corpus, is_hyperbolic=True)
            
            # Combine results
            all_results["hyperbolic"][tokenizer_type] = {
                **benchmark_results,
                **linguistic_results,
                **compression_results
            }
        except Exception as e:
            logger.error(f"Error benchmarking {tokenizer_type} hyperbolic tokenizer: {e}")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "tokenizer_comparison.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved comparison results to {results_file}")
    
    # Generate visualizations
    generate_visualizations(all_results, output_dir)


def generate_visualizations(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate visualizations from tokenizer comparison results.
    
    Args:
        results: Tokenizer comparison results
        output_dir: Directory to save visualizations
    """
    # Convert results to a DataFrame for easier plotting
    data = []
    
    # Process baseline results
    for tokenizer_name, metrics in results["baseline"].items():
        data.append({
            "tokenizer": tokenizer_name,
            "category": "baseline",
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        })
    
    # Process hyperbolic results
    for tokenizer_name, metrics in results["hyperbolic"].items():
        data.append({
            "tokenizer": tokenizer_name,
            "category": "hyperbolic",
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        })
    
    df = pd.DataFrame(data)
    
    # Set up plots
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")
    
    # Plot 1: Tokenization Speed
    plt.subplot(2, 2, 1)
    sns.barplot(x="tokenizer", y="tokens_per_second", hue="category", data=df)
    plt.title("Tokenization Speed (tokens/second)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Plot 2: Average Token Length
    plt.subplot(2, 2, 2)
    sns.barplot(x="tokenizer", y="avg_token_length", hue="category", data=df)
    plt.title("Average Token Length (characters)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Plot 3: Compression Ratio
    plt.subplot(2, 2, 3)
    sns.barplot(x="tokenizer", y="compression_ratio", hue="category", data=df)
    plt.title("Compression Ratio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Plot 4: Morpheme Ratio (Linguistic Quality)
    plt.subplot(2, 2, 4)
    sns.barplot(x="tokenizer", y="morpheme_ratio", hue="category", data=df)
    plt.title("Morpheme Boundary Ratio (Linguistic Quality)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "tokenizer_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Generate feature-specific plots for enhanced tokenizer if available
    enhanced_data = df[df["tokenizer"] == "enhanced"]
    if not enhanced_data.empty and "compression_ratio" in enhanced_data.columns and "morpheme_ratio" in enhanced_data.columns:
        plt.figure(figsize=(10, 6))
        
        # Create a radar chart for the enhanced tokenizer features
        categories = ["Tokenization Speed", "Compression Ratio", "Morpheme Ratio", "Word Boundary Ratio"]
        
        values = [
            enhanced_data["tokens_per_second"].values[0] / df["tokens_per_second"].max(),
            enhanced_data["compression_ratio"].values[0] / df["compression_ratio"].max(),
            enhanced_data["morpheme_ratio"].values[0] / df["morpheme_ratio"].max(),
            enhanced_data["word_boundary_ratio"].values[0] / df["word_boundary_ratio"].max()
        ]
        
        # Normalize between 0 and 1
        values = [max(0, min(1, v)) for v in values]
        
        # Close the plot
        values.append(values[0])
        categories.append(categories[0])
        
        # Calculate angle for each category
        angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
        
        # Plot
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories[:-1])
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
        plt.ylim(0, 1)
        
        ax.plot(angles, values)
        ax.fill(angles, values, 'b', alpha=0.1)
        
        plt.title("Enhanced Tokenizer Features (Normalized)")
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, "enhanced_tokenizer_features.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main(
    corpus_path: str = "data/processed/wiki/wiki.txt",
    baseline_dir: str = "results/baseline_tokenizers",
    standard_hyperbolic_dir: str = "results/hyperbolic/v50000",
    fast_hyperbolic_dir: str = "results/hyperbolic/fast_tokenizer",
    enhanced_hyperbolic_dir: str = "results/hyperbolic/enhanced_tokenizer",
    output_dir: str = "results/tokenizer_comparison",
    sample_size: int = 1000,
    num_runs: int = 3
) -> None:
    """
    Compare different tokenizers and generate evaluation metrics.
    
    Args:
        corpus_path: Path to corpus file
        baseline_dir: Directory containing baseline tokenizers
        standard_hyperbolic_dir: Directory containing standard hyperbolic tokenizer
        fast_hyperbolic_dir: Directory containing fast hyperbolic tokenizer
        enhanced_hyperbolic_dir: Directory containing enhanced hyperbolic tokenizer
        output_dir: Directory to save comparison results
        sample_size: Number of examples to use for evaluation
        num_runs: Number of runs for timing benchmarks
    """
    # Create hyperbolic directories dictionary
    hyperbolic_dirs = {
        "standard": standard_hyperbolic_dir,
        "fast": fast_hyperbolic_dir,
        "enhanced": enhanced_hyperbolic_dir
    }
    
    # Compare tokenizers
    compare_tokenizers(
        corpus_path=corpus_path,
        baseline_dir=baseline_dir,
        hyperbolic_dirs=hyperbolic_dirs,
        output_dir=output_dir,
        sample_size=sample_size,
        num_runs=num_runs
    )


if __name__ == "__main__":
    typer.run(main)
