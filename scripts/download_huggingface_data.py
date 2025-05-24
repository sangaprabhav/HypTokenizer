#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download datasets using Hugging Face datasets library.

This script downloads the necessary datasets for the Hyperbolic Tokenization project
using the Hugging Face datasets library, which is more reliable than direct downloads.
"""

import os
import logging
from pathlib import Path
import typer
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_wikitext(output_dir="data/processed/wikitext103"):
    """
    Download WikiText-103 dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
    """
    logger.info("Downloading WikiText-103 dataset using Hugging Face datasets")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # Save raw text
    logger.info("Saving WikiText-103 raw text")
    
    for split in ["train", "validation", "test"]:
        output_file = os.path.join(output_dir, f"{split}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                f.write(item["text"] + "\n")
        
        logger.info(f"Saved {split} set to {output_file}")


def download_yahoo_answers(output_dir="data/processed/yahoo"):
    """
    Download Yahoo! Answers Topics dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
    """
    logger.info("Downloading Yahoo! Answers Topics dataset using Hugging Face datasets")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("yahoo_answers_topics")
    
    # Save processed text
    logger.info("Saving Yahoo! Answers Topics text")
    
    for split in ["train", "test"]:
        output_file = os.path.join(output_dir, f"{split}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                f.write(f"Title: {item['question_title']}\n")
                f.write(f"Question: {item['question_content']}\n")
                f.write(f"Answer: {item['best_answer']}\n\n")
        
        logger.info(f"Saved {split} set to {output_file}")


def download_coco(output_dir="data/processed/coco"):
    """
    Generate a placeholder for MSCOCO 2017 Captions dataset.
    Since COCO isn't easily available via Hugging Face datasets,
    we'll just create a placeholder and instructions for manual download.
    
    Args:
        output_dir: Directory to save the processed dataset
    """
    logger.info("Creating placeholder for MSCOCO Captions dataset")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create placeholder file with instructions
    instructions_file = os.path.join(output_dir, "download_instructions.txt")
    
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("Instructions for downloading MSCOCO 2017 Captions dataset:\n")
        f.write("1. Visit https://cocodataset.org/#download\n")
        f.write("2. Download 2017 Train/Val annotations [241MB]\n")
        f.write("3. Download 2017 Train images [18GB] and 2017 Val images [1GB] if needed\n")
        f.write("4. Extract the downloaded files to this directory\n\n")
        f.write("Alternatively, you can use the following commands:\n")
        f.write("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n")
        f.write("unzip annotations_trainval2017.zip\n")
        f.write("wget http://images.cocodataset.org/zips/train2017.zip  # Optional, 18GB\n")
        f.write("wget http://images.cocodataset.org/zips/val2017.zip  # Optional, 1GB\n")
    
    logger.info(f"Created instructions file at {instructions_file}")
    logger.info("For tokenizer training, you can use WikiText-103 which has already been downloaded.")


def main(
    wikitext: bool = True,
    yahoo: bool = True,
    coco: bool = True,
    output_base_dir: str = "data/processed"
):
    """
    Download datasets for the Hyperbolic Tokenization project.
    
    Args:
        wikitext: Whether to download WikiText-103
        yahoo: Whether to download Yahoo! Answers Topics
        coco: Whether to download MSCOCO 2017
        output_base_dir: Base directory for processed datasets
    """
    logger.info("Starting dataset downloads using Hugging Face datasets")
    
    if wikitext:
        download_wikitext(os.path.join(output_base_dir, "wikitext103"))
    
    if yahoo:
        download_yahoo_answers(os.path.join(output_base_dir, "yahoo"))
    
    if coco:
        download_coco(os.path.join(output_base_dir, "coco"))
    
    logger.info("Dataset downloads complete")


if __name__ == "__main__":
    typer.run(main)
