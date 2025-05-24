#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and evaluate models on NLP tasks with different tokenizers.

This script trains and evaluates models for masked language modeling (MLM)
and text classification using different tokenization methods.
"""

import torch
import random
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import typer
from pathlib import Path
import sys
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    BertConfig, BertForMaskedLM, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# Add parent directory to path to import from embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.poincare_ball import log_map_zero


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
    
    # Set transformers specific seed
    import transformers
    transformers.set_seed(seed)


class SimpleTokenizer:
    """
    A simple wrapper around different tokenizers to provide a common interface.
    """
    
    def __init__(self, method: str, vocab_size: int, model_path: str):
        """
        Initialize the tokenizer.
        
        Args:
            method: Tokenization method ('bpe', 'wordpiece', 'unigram', or 'hyperbolic')
            vocab_size: Vocabulary size
            model_path: Path to tokenizer model
        """
        self.method = method
        self.vocab_size = vocab_size
        self.model_path = model_path
        
        if method == "hyperbolic":
            # Load hyperbolic tokenizer
            from tokenizer.hyperbolic_merge import HyperbolicTokenizer
            self.tokenizer = HyperbolicTokenizer.load(model_path)
        else:
            # Load SentencePiece tokenizer
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_path}/{method}/v{vocab_size}.model")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.method == "hyperbolic":
            return self.tokenizer.tokenize(text)
        else:
            return self.sp.encode_as_pieces(text)
    
    def encode(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Encode text for transformer models.
        
        Args:
            text: Input text
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if self.method == "hyperbolic":
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.sp.encode(text, out_type=int)
        
        # Truncate if needed
        max_length = kwargs.get("max_length", 128)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        return {"input_ids": tokens, "attention_mask": attention_mask}
    
    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded = self.encode(text, **kwargs)
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        
        return {"input_ids": input_ids, "attention_mask": attention_masks}
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        if self.method == "hyperbolic":
            return len(self.tokenizer.vocab)
        else:
            return self.sp.get_piece_size()
    
    def get_embeddings(self) -> torch.Tensor:
        """
        Get token embeddings.
        
        Returns:
            Embeddings tensor
        """
        if self.method == "hyperbolic":
            # Convert from Lorentz to Euclidean
            lorentz_emb = self.tokenizer.embeddings.detach()
            return log_map_zero(lorentz_emb)
        else:
            # For baseline tokenizers, we'll return None and let the caller initialize them
            return None


def load_wikitext_dataset():
    """
    Load WikiText-103 dataset for MLM.
    
    Returns:
        WikiText-103 dataset
    """
    logger.info("Loading WikiText-103 dataset")
    return load_dataset("wikitext", "wikitext-103-raw-v1")


def load_yahoo_dataset():
    """
    Load Yahoo! Answers Topics dataset for classification.
    
    Returns:
        Yahoo! Answers Topics dataset
    """
    logger.info("Loading Yahoo! Answers Topics dataset")
    return load_dataset("yahoo_answers_topics")


def tokenize_function(examples, tokenizer, max_length=128):
    """
    Tokenize examples for transformer models.
    
    Args:
        examples: Examples to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer.batch_encode(examples["text"], max_length=max_length, truncation=True)


def train_mlm(
    tokenizer,
    dataset,
    output_dir: str,
    config: BertConfig,
    training_args: TrainingArguments,
    device: torch.device
) -> Dict[str, float]:
    """
    Train a masked language model.
    
    Args:
        tokenizer: Tokenizer to use
        dataset: Dataset to train on
        output_dir: Directory to save model
        config: Model configuration
        training_args: Training arguments
        device: Device to use
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Training MLM with {tokenizer.method} tokenizer (V={tokenizer.vocab_size})")
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Initialize model
    model = BertForMaskedLM(config)
    model.to(device)
    
    # Get tokenizer embeddings
    embeddings = tokenizer.get_embeddings()
    if embeddings is not None:
        logger.info(f"Initializing model embeddings from tokenizer (shape={embeddings.shape})")
        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(embeddings)
    
    # Create data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model")
    eval_results = trainer.evaluate()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/mlm_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    
    return eval_results


def train_classification(
    tokenizer,
    dataset,
    output_dir: str,
    config: BertConfig,
    training_args: TrainingArguments,
    device: torch.device
) -> Dict[str, float]:
    """
    Train a text classification model.
    
    Args:
        tokenizer: Tokenizer to use
        dataset: Dataset to train on
        output_dir: Directory to save model
        config: Model configuration
        training_args: Training arguments
        device: Device to use
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Training classification with {tokenizer.method} tokenizer (V={tokenizer.vocab_size})")
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["question_title", "question_content", "best_answer"]
    )
    
    # Initialize model
    model = BertForSequenceClassification(config)
    model.to(device)
    
    # Get tokenizer embeddings
    embeddings = tokenizer.get_embeddings()
    if embeddings is not None:
        logger.info(f"Initializing model embeddings from tokenizer (shape={embeddings.shape})")
        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(embeddings)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model")
    eval_results = trainer.evaluate()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/classification_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    
    return eval_results


def train_tasks(
    method: str,
    vocab_size: int,
    model_path: str,
    output_dir: str,
    embedding_dim: int = 768,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    mlm_only: bool = False,
    seed: int = 42
) -> None:
    """
    Train models on NLP tasks with a specific tokenizer.
    
    Args:
        method: Tokenization method ('bpe', 'wordpiece', 'unigram', or 'hyperbolic')
        vocab_size: Vocabulary size
        model_path: Path to tokenizer model
        output_dir: Directory to save results
        embedding_dim: Embedding dimension
        num_hidden_layers: Number of hidden layers
        num_attention_heads: Number of attention heads
        mlm_only: Whether to train only the MLM model
        seed: Random seed
    """
    # Set seeds
    set_seeds(seed)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(method, vocab_size, model_path)
    
    # Create model configuration
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=embedding_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=embedding_dim * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute"
    )
    
    # Train MLM
    mlm_dataset = load_wikitext_dataset()
    
    mlm_training_args = TrainingArguments(
        output_dir=f"{output_dir}/mlm",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_steps=500,
        save_steps=1000,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=1,
    )
    
    mlm_results = train_mlm(
        tokenizer=tokenizer,
        dataset=mlm_dataset,
        output_dir=output_dir,
        config=config,
        training_args=mlm_training_args,
        device=device
    )
    
    # Train classification if requested
    if not mlm_only:
        classification_dataset = load_yahoo_dataset()
        
        # Update config for classification
        config.num_labels = 10  # Yahoo has 10 classes
        
        classification_training_args = TrainingArguments(
            output_dir=f"{output_dir}/classification",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            logging_steps=500,
            save_steps=1000,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            weight_decay=0.01,
            save_total_limit=1,
        )
        
        classification_results = train_classification(
            tokenizer=tokenizer,
            dataset=classification_dataset,
            output_dir=output_dir,
            config=config,
            training_args=classification_training_args,
            device=device
        )


def main(
    method: str = "hyperbolic",
    vocab_size: int = 50000,
    model_path: str = "results/hyperbolic/v50000",
    output_dir: str = "results/hyperbolic/v50000/tasks",
    embedding_dim: int = 768,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    mlm_only: bool = False,
    seed: int = 42
) -> None:
    """
    Train models on NLP tasks with a specific tokenizer.
    """
    train_tasks(
        method=method,
        vocab_size=vocab_size,
        model_path=model_path,
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        mlm_only=mlm_only,
        seed=seed
    )


if __name__ == "__main__":
    typer.run(main)
