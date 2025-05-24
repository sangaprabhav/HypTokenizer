#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess Wikipedia data for tokenizer training.

This script processes raw Wikipedia dumps into a clean text format
suitable for training tokenizers.
"""

import re
import unicodedata
import os
import glob
import bz2
from typing import List, Optional, Union, TextIO
import typer
from pathlib import Path
from tqdm import tqdm
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by normalizing and removing special characters.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Remove special characters, keeping only alphanumeric, spaces, periods, and commas
    text = re.sub(r'[^a-z0-9\s\.\,]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def open_file(file_path: str, mode: str = 'r') -> Union[TextIO, bz2.BZ2File]:
    """
    Open a file, automatically handling BZ2 compressed files.
    
    Args:
        file_path: Path to the file
        mode: Open mode ('r' or 'w')
        
    Returns:
        File object
    """
    if file_path.endswith('.bz2'):
        if 'r' in mode:
            return bz2.open(file_path, mode + 't', encoding='utf-8', errors='ignore')
        else:
            return bz2.open(file_path, mode + 't', encoding='utf-8')
    else:
        if 'r' in mode:
            return open(file_path, mode, encoding='utf-8', errors='ignore')
        else:
            return open(file_path, mode, encoding='utf-8')


def process_wiki_files(input_path: str, output_file: str) -> None:
    """
    Process Wikipedia files (raw or BZ2 compressed) and write to a single output file.
    
    Args:
        input_path: Path to input file (BZ2) or directory containing extracted files
        output_file: Path to output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        # Check if input_path is a file or directory
        if os.path.isfile(input_path):
            # Single file (likely BZ2)
            logger.info(f"Processing single file: {input_path}")
            
            try:
                with open_file(input_path, "r") as in_f:
                    for i, line in enumerate(tqdm(in_f, desc="Processing Wiki content")):
                        # Log progress periodically for large files
                        if i > 0 and i % 100000 == 0:
                            logger.info(f"Processed {i} lines")
                        
                        cleaned = clean_text(line)
                        if cleaned:
                            out_f.write(cleaned + " \n")
            except Exception as e:
                logger.error(f"Error processing file {input_path}: {e}")
        else:
            # Directory with multiple files
            files = sorted(glob.glob(os.path.join(input_path, "*/*")))
            logger.info(f"Found {len(files)} files to process")
            
            # Process files
            for file_path in tqdm(files, desc="Processing Wiki files"):
                try:
                    with open_file(file_path, "r") as in_f:
                        for line in in_f:
                            cleaned = clean_text(line)
                            if cleaned:
                                out_f.write(cleaned + " \n")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
    
    logger.info(f"Completed processing. Output written to {output_file}")


def build_initial_vocab(input_file: str, output_file: str, min_count: int = 5) -> None:
    """
    Build initial vocabulary from processed Wikipedia text.
    
    Args:
        input_file: Path to processed Wikipedia text
        output_file: Path to output vocabulary file
        min_count: Minimum count for a token to be included
    """
    # Count character frequencies
    char_counts = {}
    
    logger.info(f"Building initial vocabulary from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Counting characters")):
            for char in line:
                if char in char_counts:
                    char_counts[char] += 1
                else:
                    char_counts[char] = 1
            
            # Log progress
            if i > 0 and i % 1000000 == 0:
                logger.info(f"Processed {i} lines")
    
    # Filter by minimum count
    vocab = [char for char, count in char_counts.items() if count >= min_count]
    
    # Add special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = special_tokens + vocab
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write vocabulary to file
    with open(output_file, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    
    logger.info(f"Wrote vocabulary with {len(vocab)} tokens to {output_file}")


def main(
    input_path: str = "/Users/prabhavsanga/Desktop/HypTokenizer/data/raw/enwiki-latest-pages-articles-multistream-index.txt.bz2",
    output_file: str = "data/processed/wiki/wiki.txt",
    vocab_file: str = "data/processed/wiki/vocab_initial.txt",
    min_count: int = 5
) -> None:
    """
    Process Wikipedia data and build initial vocabulary.
    
    Args:
        input_path: Path to input file (BZ2) or directory containing extracted files
        output_file: Path to output processed text file
        vocab_file: Path to output vocabulary file
        min_count: Minimum count for a token to be included in vocabulary
    """
    # Process Wikipedia files
    process_wiki_files(input_path, output_file)
    
    # Build initial vocabulary
    build_initial_vocab(output_file, vocab_file, min_count)


if __name__ == "__main__":
    typer.run(main)
