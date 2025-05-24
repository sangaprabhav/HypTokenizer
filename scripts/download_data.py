#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download data for the Hyperbolic Tokenization project.

This script downloads the necessary datasets using Python libraries
rather than relying on external commands like wget.
"""

import os
import urllib.request
import zipfile
import bz2
import shutil
from pathlib import Path
from tqdm import tqdm
import typer
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc=None):
    """
    Download a file from a URL showing progress.
    
    Args:
        url: URL to download
        output_path: Path to save the file
        desc: Description for the progress bar
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not desc:
        desc = os.path.basename(output_path)
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return
    
    logger.info(f"Downloading {url} to {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def extract_bz2(bz2_path, extract_path):
    """
    Extract a BZ2 file.
    
    Args:
        bz2_path: Path to the BZ2 file
        extract_path: Path to extract to
    """
    logger.info(f"Extracting {bz2_path} to {extract_path}")
    
    os.makedirs(os.path.dirname(extract_path), exist_ok=True)
    
    with open(extract_path, 'wb') as new_file, bz2.BZ2File(bz2_path, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)


def download_wikipedia(output_dir="data/raw"):
    """
    Download Wikipedia dump.
    
    Args:
        output_dir: Directory to save the dump
    """
    url = "https://dumps.wikimedia.org/enwiki/20220301/enwiki-20220301-pages-articles.xml.bz2"
    output_path = os.path.join(output_dir, "enwiki-20220301-pages-articles.xml.bz2")
    
    download_url(url, output_path, "Wikipedia dump")


def download_wikitext(output_dir="data/raw"):
    """
    Download WikiText-103.
    
    Args:
        output_dir: Directory to save the dataset
    """
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    output_path = os.path.join(output_dir, "wikitext-103-v1.zip")
    
    download_url(url, output_path, "WikiText-103")
    
    # Extract
    extract_dir = os.path.join(output_dir, "../processed/wikitext103")
    extract_zip(output_path, extract_dir)


def download_coco(output_dir="data/raw"):
    """
    Download MSCOCO 2017 Captions.
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Train images
    train_url = "http://images.cocodataset.org/zips/train2017.zip"
    train_path = os.path.join(output_dir, "train2017.zip")
    download_url(train_url, train_path, "COCO Train Images")
    
    # Validation images
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_path = os.path.join(output_dir, "val2017.zip")
    download_url(val_url, val_path, "COCO Val Images")
    
    # Annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_path = os.path.join(output_dir, "annotations_trainval2017.zip")
    download_url(ann_url, ann_path, "COCO Annotations")
    
    # Extract
    extract_img_dir = os.path.join(output_dir, "../processed/coco/images")
    extract_ann_dir = os.path.join(output_dir, "../processed/coco/annotations")
    
    os.makedirs(extract_img_dir, exist_ok=True)
    os.makedirs(extract_ann_dir, exist_ok=True)
    
    extract_zip(train_path, extract_img_dir)
    extract_zip(val_path, extract_img_dir)
    extract_zip(ann_path, extract_ann_dir)


def main(
    wikipedia: bool = False,
    wikitext: bool = True,
    coco: bool = True,
    output_dir: str = "data/raw"
):
    """
    Download datasets for the Hyperbolic Tokenization project.
    
    Args:
        wikipedia: Whether to download Wikipedia dump
        wikitext: Whether to download WikiText-103
        coco: Whether to download MSCOCO 2017
        output_dir: Directory to save the datasets
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download datasets
    if wikipedia:
        download_wikipedia(output_dir)
    
    if wikitext:
        download_wikitext(output_dir)
    
    if coco:
        download_coco(output_dir)


if __name__ == "__main__":
    typer.run(main)
