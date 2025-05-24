#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark tokenization efficiency for different tokenizers.

This script measures the tokenization throughput and training time
for different tokenization methods.
"""

import torch
import time
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import typer
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import from tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_lines(file_path: str, num_lines: int = 1000000) -> List[str]:
    """
    Load lines from a file.
    
    Args:
        file_path: Path to the file
        num_lines: Number of lines to load
        
    Returns:
        List of lines
    """
    logger.info(f"Loading {num_lines} lines from {file_path}")
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            lines.append(line.strip())
    
    logger.info(f"Loaded {len(lines)} lines")
    return lines


def benchmark_tokenization_throughput(
    tokenizer,
    lines: List[str],
    warmup: int = 1000
) -> float:
    """
    Benchmark tokenization throughput.
    
    Args:
        tokenizer: Tokenizer to benchmark
        lines: List of lines to tokenize
        warmup: Number of warmup iterations
        
    Returns:
        Tokenization throughput (tokens per second)
    """
    # Warmup
    for i in range(warmup):
        _ = tokenizer.tokenize(lines[i % len(lines)])
    
    # Benchmark
    start_time = time.perf_counter()
    total_tokens = 0
    
    for line in tqdm(lines, desc="Benchmarking tokenization"):
        tokens = tokenizer.tokenize(line)
        total_tokens += len(tokens)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    throughput = total_tokens / elapsed_time
    
    logger.info(f"Tokenized {total_tokens} tokens in {elapsed_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} tokens/second")
    
    return throughput


class SentencePieceWrapper:
    """
    Wrapper for SentencePiece tokenizers.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tokenizer.
        
        Args:
            model_path: Path to the SentencePiece model
        """
        import sentencepiece as spm
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.encode_as_pieces(text)


def benchmark_tokenizers(
    hyperbolic_path: str,
    bpe_path: str,
    wordpiece_path: str,
    unigram_path: str,
    text_path: str,
    output_path: str,
    num_lines: int = 100000
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different tokenizers.
    
    Args:
        hyperbolic_path: Path to hyperbolic tokenizer
        bpe_path: Path to BPE tokenizer
        wordpiece_path: Path to WordPiece tokenizer
        unigram_path: Path to Unigram tokenizer
        text_path: Path to text file
        output_path: Path to save results
        num_lines: Number of lines to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    # Load lines
    lines = load_lines(text_path, num_lines)
    
    # Initialize tokenizers
    tokenizers = {}
    
    # Hyperbolic tokenizer
    logger.info("Loading hyperbolic tokenizer")
    tokenizers["hyperbolic"] = HyperbolicTokenizer.load(hyperbolic_path)
    
    # BPE tokenizer
    logger.info("Loading BPE tokenizer")
    tokenizers["bpe"] = SentencePieceWrapper(bpe_path)
    
    # WordPiece tokenizer
    logger.info("Loading WordPiece tokenizer")
    tokenizers["wordpiece"] = SentencePieceWrapper(wordpiece_path)
    
    # Unigram tokenizer
    logger.info("Loading Unigram tokenizer")
    tokenizers["unigram"] = SentencePieceWrapper(unigram_path)
    
    # Benchmark tokenizers
    results = {}
    
    for name, tokenizer in tokenizers.items():
        logger.info(f"Benchmarking {name} tokenizer")
        throughput = benchmark_tokenization_throughput(tokenizer, lines)
        results[name] = {"throughput": throughput}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved benchmark results to {output_path}")
    
    return results


def benchmark_training_time(
    training_log_paths: Dict[str, str],
    output_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark training time for different tokenizers.
    
    Args:
        training_log_paths: Dictionary mapping tokenizer names to log file paths
        output_path: Path to save results
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for name, log_path in training_log_paths.items():
        logger.info(f"Processing training log for {name}")
        
        # Load training log
        with open(log_path, "r") as f:
            log_data = json.load(f)
        
        # Extract training time
        if "training_time" in log_data:
            training_time = log_data["training_time"]
        else:
            # Estimate from timestamps if available
            training_time = None
            if "timestamps" in log_data:
                timestamps = log_data["timestamps"]
                training_time = timestamps[-1] - timestamps[0]
        
        results[name] = {"training_time": training_time}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved training time results to {output_path}")
    
    return results


def benchmark_efficiency(
    hyperbolic_path: str = "results/hyperbolic/v50000",
    bpe_path: str = "data/processed/tokenizers/bpe/v50000.model",
    wordpiece_path: str = "data/processed/tokenizers/wordpiece/v50000.model",
    unigram_path: str = "data/processed/tokenizers/unigram/v50000.model",
    text_path: str = "data/processed/wiki/wiki.txt",
    output_dir: str = "results/efficiency",
    num_lines: int = 100000
) -> None:
    """
    Benchmark tokenization efficiency.
    
    Args:
        hyperbolic_path: Path to hyperbolic tokenizer
        bpe_path: Path to BPE tokenizer
        wordpiece_path: Path to WordPiece tokenizer
        unigram_path: Path to Unigram tokenizer
        text_path: Path to text file
        output_dir: Directory to save results
        num_lines: Number of lines to benchmark
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark tokenization throughput
    throughput_results = benchmark_tokenizers(
        hyperbolic_path=hyperbolic_path,
        bpe_path=bpe_path,
        wordpiece_path=wordpiece_path,
        unigram_path=unigram_path,
        text_path=text_path,
        output_path=f"{output_dir}/throughput.json",
        num_lines=num_lines
    )
    
    # Check if training logs exist
    training_log_paths = {
        "hyperbolic": f"{hyperbolic_path}/training_stats.json",
    }
    
    # Check for baseline training logs
    for method in ["bpe", "wordpiece", "unigram"]:
        log_path = f"data/processed/tokenizers/{method}/training_stats.json"
        if os.path.exists(log_path):
            training_log_paths[method] = log_path
    
    # Benchmark training time if logs exist
    if all(os.path.exists(path) for path in training_log_paths.values()):
        training_time_results = benchmark_training_time(
            training_log_paths=training_log_paths,
            output_path=f"{output_dir}/training_time.json"
        )
    else:
        logger.warning("Some training logs are missing, skipping training time benchmark")


def main(
    hyperbolic_path: str = "results/hyperbolic/v50000",
    bpe_path: str = "data/processed/tokenizers/bpe/v50000.model",
    wordpiece_path: str = "data/processed/tokenizers/wordpiece/v50000.model",
    unigram_path: str = "data/processed/tokenizers/unigram/v50000.model",
    text_path: str = "data/processed/wiki/wiki.txt",
    output_dir: str = "results/efficiency",
    num_lines: int = 100000
) -> None:
    """
    Benchmark tokenization efficiency.
    """
    benchmark_efficiency(
        hyperbolic_path=hyperbolic_path,
        bpe_path=bpe_path,
        wordpiece_path=wordpiece_path,
        unigram_path=unigram_path,
        text_path=text_path,
        output_dir=output_dir,
        num_lines=num_lines
    )


if __name__ == "__main__":
    typer.run(main)
