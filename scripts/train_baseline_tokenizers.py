#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train baseline tokenizers for comparison.

This script trains standard tokenizers (BPE, WordPiece, Unigram) using
the Hugging Face tokenizers library to provide baselines for comparison 
with the hyperbolic tokenizers.
"""

import os
import json
import random
import logging
import numpy as np
import torch
import typer
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import time
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
import sys

# Add parent directory to path to import from tokenizer and embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.fast_hyperbolic_merge import FastHyperbolicTokenizer

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


def load_corpus(corpus_path: str) -> List[str]:
    """
    Load training corpus.
    
    Args:
        corpus_path: Path to corpus file
        
    Returns:
        List of training examples
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_bpe_tokenizer(
    corpus_path: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: List[str] = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
) -> Tuple[Tokenizer, Dict[str, Any]]:
    """
    Train a BPE tokenizer.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Directory to save the tokenizer
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens
        
    Returns:
        Trained tokenizer and training statistics
    """
    start_time = time.time()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Initialize normalizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    # Initialize pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train
    logger.info(f"Training BPE tokenizer with vocab size {vocab_size}")
    tokenizer.train([corpus_path], trainer)
    
    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Calculate stats
    end_time = time.time()
    training_time = end_time - start_time
    vocab = tokenizer.get_vocab()
    actual_vocab_size = len(vocab)
    
    # Encode sample text for metrics
    sample_text = load_corpus(corpus_path)[:1000]  # Take first 1000 lines for evaluation
    avg_token_length = 0
    avg_tokens_per_line = 0
    
    for line in sample_text:
        encoding = tokenizer.encode(line)
        tokens = encoding.tokens
        avg_tokens_per_line += len(tokens)
        for token in tokens:
            avg_token_length += len(token)
    
    avg_tokens_per_line /= len(sample_text)
    avg_token_length /= (avg_tokens_per_line * len(sample_text))
    
    # Save stats
    stats = {
        "tokenizer_type": "BPE",
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "training_time_seconds": training_time,
        "avg_token_length": avg_token_length,
        "avg_tokens_per_line": avg_tokens_per_line,
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved BPE tokenizer to {tokenizer_path}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Vocabulary size: {actual_vocab_size}")
    
    return tokenizer, stats


def train_wordpiece_tokenizer(
    corpus_path: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: List[str] = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
) -> Tuple[Tokenizer, Dict[str, Any]]:
    """
    Train a WordPiece tokenizer.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Directory to save the tokenizer
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens
        
    Returns:
        Trained tokenizer and training statistics
    """
    start_time = time.time()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    # Initialize normalizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    # Initialize pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train
    logger.info(f"Training WordPiece tokenizer with vocab size {vocab_size}")
    tokenizer.train([corpus_path], trainer)
    
    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Calculate stats
    end_time = time.time()
    training_time = end_time - start_time
    vocab = tokenizer.get_vocab()
    actual_vocab_size = len(vocab)
    
    # Encode sample text for metrics
    sample_text = load_corpus(corpus_path)[:1000]  # Take first 1000 lines for evaluation
    avg_token_length = 0
    avg_tokens_per_line = 0
    
    for line in sample_text:
        encoding = tokenizer.encode(line)
        tokens = encoding.tokens
        avg_tokens_per_line += len(tokens)
        for token in tokens:
            avg_token_length += len(token)
    
    avg_tokens_per_line /= len(sample_text)
    avg_token_length /= (avg_tokens_per_line * len(sample_text))
    
    # Save stats
    stats = {
        "tokenizer_type": "WordPiece",
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "training_time_seconds": training_time,
        "avg_token_length": avg_token_length,
        "avg_tokens_per_line": avg_tokens_per_line,
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved WordPiece tokenizer to {tokenizer_path}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Vocabulary size: {actual_vocab_size}")
    
    return tokenizer, stats


def train_unigram_tokenizer(
    corpus_path: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: List[str] = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
) -> Tuple[Tokenizer, Dict[str, Any]]:
    """
    Train a Unigram tokenizer.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Directory to save the tokenizer
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens
        
    Returns:
        Trained tokenizer and training statistics
    """
    start_time = time.time()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(Unigram())
    
    # Initialize normalizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    # Initialize pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize trainer
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="<unk>",
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train
    logger.info(f"Training Unigram tokenizer with vocab size {vocab_size}")
    tokenizer.train([corpus_path], trainer)
    
    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Calculate stats
    end_time = time.time()
    training_time = end_time - start_time
    vocab = tokenizer.get_vocab()
    actual_vocab_size = len(vocab)
    
    # Encode sample text for metrics
    sample_text = load_corpus(corpus_path)[:1000]  # Take first 1000 lines for evaluation
    avg_token_length = 0
    avg_tokens_per_line = 0
    
    for line in sample_text:
        encoding = tokenizer.encode(line)
        tokens = encoding.tokens
        avg_tokens_per_line += len(tokens)
        for token in tokens:
            avg_token_length += len(token)
    
    avg_tokens_per_line /= len(sample_text)
    avg_token_length /= (avg_tokens_per_line * len(sample_text))
    
    # Save stats
    stats = {
        "tokenizer_type": "Unigram",
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "training_time_seconds": training_time,
        "avg_token_length": avg_token_length,
        "avg_tokens_per_line": avg_tokens_per_line,
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved Unigram tokenizer to {tokenizer_path}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Vocabulary size: {actual_vocab_size}")
    
    return tokenizer, stats


def train_character_tokenizer(
    corpus_path: str,
    output_dir: str,
    special_tokens: List[str] = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
) -> Tuple[Tokenizer, Dict[str, Any]]:
    """
    Create a character-level tokenizer.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Directory to save the tokenizer
        special_tokens: List of special tokens
        
    Returns:
        Trained tokenizer and training statistics
    """
    start_time = time.time()
    
    # Initialize tokenizer with BPE but use character-level pre-tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Initialize normalizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    # Initialize character-level pre-tokenizer
    tokenizer.pre_tokenizer = CharDelimiterSplit("")
    
    # Read corpus and get unique characters
    unique_chars = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            for char in line.strip():
                unique_chars.add(char)
    
    # Add special tokens
    for token in special_tokens:
        unique_chars.add(token)
    
    # Create a vocabulary
    vocab = {char: i for i, char in enumerate(sorted(unique_chars))}
    
    # Set vocabulary directly (no training needed for character-level)
    tokenizer.model = BPE(vocab, {}, unk_token="[UNK]")
    
    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Calculate stats
    end_time = time.time()
    training_time = end_time - start_time
    actual_vocab_size = len(vocab)
    
    # Encode sample text for metrics
    sample_text = load_corpus(corpus_path)[:1000]  # Take first 1000 lines for evaluation
    avg_token_length = 0
    avg_tokens_per_line = 0
    
    for line in sample_text:
        encoding = tokenizer.encode(line)
        tokens = encoding.tokens
        avg_tokens_per_line += len(tokens)
        for token in tokens:
            avg_token_length += len(token)
    
    avg_tokens_per_line /= len(sample_text)
    avg_token_length /= (avg_tokens_per_line * len(sample_text))
    
    # Save stats
    stats = {
        "tokenizer_type": "Character",
        "vocab_size": actual_vocab_size,
        "training_time_seconds": training_time,
        "avg_token_length": avg_token_length,
        "avg_tokens_per_line": avg_tokens_per_line,
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved Character tokenizer to {tokenizer_path}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Vocabulary size: {actual_vocab_size}")
    
    return tokenizer, stats


def compare_with_hyperbolic(
    corpus_path: str,
    hyperbolic_model_path: str,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Compare metrics with hyperbolic tokenizer.
    
    Args:
        corpus_path: Path to corpus file
        hyperbolic_model_path: Path to hyperbolic tokenizer model
        sample_size: Number of lines to sample for comparison
        
    Returns:
        Comparison metrics
    """
    logger.info(f"Loading hyperbolic tokenizer from {hyperbolic_model_path}")
    hyperbolic_tokenizer = FastHyperbolicTokenizer.load(hyperbolic_model_path)
    
    sample_text = load_corpus(corpus_path)[:sample_size]
    
    # Compute metrics for hyperbolic tokenizer
    hyp_avg_token_length = 0
    hyp_avg_tokens_per_line = 0
    hyp_tokenization_time = 0
    
    start_time = time.time()
    for line in sample_text:
        tokens = hyperbolic_tokenizer.tokenize(line)
        hyp_avg_tokens_per_line += len(tokens)
        for token in tokens:
            hyp_avg_token_length += len(token)
    end_time = time.time()
    
    hyp_tokenization_time = end_time - start_time
    hyp_avg_tokens_per_line /= len(sample_text)
    hyp_avg_token_length /= (hyp_avg_tokens_per_line * len(sample_text))
    
    metrics = {
        "hyperbolic_vocab_size": len(hyperbolic_tokenizer.vocab),
        "hyperbolic_avg_token_length": hyp_avg_token_length,
        "hyperbolic_avg_tokens_per_line": hyp_avg_tokens_per_line,
        "hyperbolic_tokenization_time": hyp_tokenization_time,
        "sample_size": sample_size,
    }
    
    return metrics


def train_all_tokenizers(
    corpus_path: str,
    output_dir: str,
    vocab_sizes: List[int] = [10000, 20000, 50000],
    seed: int = 42,
    hyperbolic_model_path: Optional[str] = None,
) -> None:
    """
    Train all types of tokenizers with various vocab sizes.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Base directory to save tokenizers
        vocab_sizes: List of vocabulary sizes to train
        seed: Random seed
        hyperbolic_model_path: Path to hyperbolic tokenizer model for comparison
    """
    set_seeds(seed)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Train character-level tokenizer (fixed vocab size)
    char_dir = os.path.join(output_dir, "character")
    train_character_tokenizer(corpus_path, char_dir)
    
    # Train tokenizers for each type and vocab size
    tokenizer_types = ["bpe", "wordpiece", "unigram"]
    all_stats = {}
    
    for tokenizer_type in tokenizer_types:
        all_stats[tokenizer_type] = {}
        
        for vocab_size in vocab_sizes:
            type_dir = os.path.join(output_dir, f"{tokenizer_type}_v{vocab_size}")
            
            if tokenizer_type == "bpe":
                _, stats = train_bpe_tokenizer(corpus_path, type_dir, vocab_size)
            elif tokenizer_type == "wordpiece":
                _, stats = train_wordpiece_tokenizer(corpus_path, type_dir, vocab_size)
            elif tokenizer_type == "unigram":
                _, stats = train_unigram_tokenizer(corpus_path, type_dir, vocab_size)
            
            all_stats[tokenizer_type][vocab_size] = stats
    
    # Compare with hyperbolic tokenizer if path provided
    if hyperbolic_model_path:
        comparison = compare_with_hyperbolic(corpus_path, hyperbolic_model_path)
        all_stats["comparison"] = comparison
    
    # Save all stats
    with open(os.path.join(output_dir, "all_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"Saved all tokenizer statistics to {output_dir}/all_stats.json")


def main(
    corpus_path: str = "data/processed/wiki/wiki.txt",
    output_dir: str = "results/baseline_tokenizers",
    vocab_sizes: str = "10000,20000,50000",
    tokenizer_type: str = "all",
    vocab_size: int = 50000,
    seed: int = 42,
    hyperbolic_model_path: Optional[str] = None,
) -> None:
    """
    Train baseline tokenizers with the given parameters.
    
    Args:
        corpus_path: Path to corpus file
        output_dir: Directory to save tokenizers
        vocab_sizes: Comma-separated list of vocabulary sizes for training all tokenizers
        tokenizer_type: Type of tokenizer to train ('bpe', 'wordpiece', 'unigram', 'character', 'all')
        vocab_size: Target vocabulary size when training a single tokenizer type
        seed: Random seed
        hyperbolic_model_path: Path to hyperbolic tokenizer model for comparison
    """
    set_seeds(seed)
    
    # Parse vocab sizes
    sizes = [int(s.strip()) for s in vocab_sizes.split(",")]
    
    if tokenizer_type == "all":
        train_all_tokenizers(corpus_path, output_dir, sizes, seed, hyperbolic_model_path)
    else:
        # Train specific tokenizer type
        type_dir = os.path.join(output_dir, f"{tokenizer_type}_v{vocab_size}")
        os.makedirs(type_dir, exist_ok=True)
        
        if tokenizer_type == "bpe":
            train_bpe_tokenizer(corpus_path, type_dir, vocab_size)
        elif tokenizer_type == "wordpiece":
            train_wordpiece_tokenizer(corpus_path, type_dir, vocab_size)
        elif tokenizer_type == "unigram":
            train_unigram_tokenizer(corpus_path, type_dir, vocab_size)
        elif tokenizer_type == "character":
            train_character_tokenizer(corpus_path, type_dir)
        else:
            logger.error(f"Unknown tokenizer type: {tokenizer_type}")
            sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
