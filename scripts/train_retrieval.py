#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a cross-modal retrieval model using hyperbolic embeddings.

This script trains a two-tower model for text-image retrieval using
hyperbolic embeddings and contrastive learning.
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
from transformers import ViTFeatureExtractor, ViTModel, BertModel, BertTokenizer

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.lorentz_model import exp_map, distance, project_to_hyperboloid
from multimodal.contrastive_loss import hyperbolic_contrastive_loss, MultimodalHyperbolicModel


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


class CocoDataset(torch.utils.data.Dataset):
    """
    Dataset class for COCO Captions.
    """
    
    def __init__(
        self,
        dataset,
        image_processor,
        text_tokenizer,
        split: str = "train",
        max_length: int = 64
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: HuggingFace COCO dataset
            image_processor: Image processor
            text_tokenizer: Text tokenizer
            split: Dataset split ('train' or 'validation')
            max_length: Maximum text length
        """
        self.dataset = dataset[split]
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # Process image
        image = item["image"]
        image_inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"][0]
        
        # Process text
        caption = item["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)  # Randomly select one caption if there are multiple
        
        text_inputs = self.text_tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Return processed inputs
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"][0],
            "attention_mask": text_inputs["attention_mask"][0],
            "caption": caption
        }


class TextEncoder(torch.nn.Module):
    """
    Text encoder for the retrieval model.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Name of the pre-trained model
        """
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Text features
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output


class ImageEncoder(torch.nn.Module):
    """
    Image encoder for the retrieval model.
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the image encoder.
        
        Args:
            model_name: Name of the pre-trained model
        """
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
    
    def forward(self, pixel_values):
        """
        Forward pass through the image encoder.
        
        Args:
            pixel_values: Pixel values
            
        Returns:
            Image features
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output


def compute_recall_at_k(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K for text-to-image and image-to-text retrieval.
    
    Args:
        text_embeddings: Text embeddings
        image_embeddings: Image embeddings
        k_values: K values for Recall@K
        
    Returns:
        Dictionary with Recall@K values
    """
    batch_size = text_embeddings.size(0)
    results = {}
    
    # Compute pairwise distances
    distances = torch.zeros((batch_size, batch_size), device=text_embeddings.device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            distances[i, j] = distance(
                text_embeddings[i].unsqueeze(0),
                image_embeddings[j].unsqueeze(0)
            ).item()
    
    # Text-to-image retrieval
    for k in k_values:
        correct = 0
        for i in range(batch_size):
            # Get top-k closest images
            _, indices = torch.topk(distances[i], k, largest=False)
            if i in indices:
                correct += 1
        
        recall = correct / batch_size
        results[f"r@{k}_text2image"] = recall
    
    # Image-to-text retrieval
    for k in k_values:
        correct = 0
        for j in range(batch_size):
            # Get top-k closest texts
            _, indices = torch.topk(distances[:, j], k, largest=False)
            if j in indices:
                correct += 1
        
        recall = correct / batch_size
        results[f"r@{k}_image2text"] = recall
    
    return results


def train_retrieval(
    output_dir: str,
    embedding_dim: int = 64,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 2e-5,
    temperature: float = 0.07,
    curvature: float = 1.0,
    seed: int = 42
) -> Dict[str, float]:
    """
    Train a cross-modal retrieval model.
    
    Args:
        output_dir: Directory to save results
        embedding_dim: Dimension of the embedding space
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature parameter for contrastive loss
        curvature: Curvature parameter
        seed: Random seed
        
    Returns:
        Dictionary with evaluation results
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
    
    # Load COCO dataset
    logger.info("Loading COCO Captions dataset")
    coco_dataset = load_dataset("coco_captions", "2017")
    
    # Initialize image processor and text tokenizer
    logger.info("Initializing image processor and text tokenizer")
    image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset and data loader
    train_dataset = CocoDataset(
        dataset=coco_dataset,
        image_processor=image_processor,
        text_tokenizer=text_tokenizer,
        split="train"
    )
    
    val_dataset = CocoDataset(
        dataset=coco_dataset,
        image_processor=image_processor,
        text_tokenizer=text_tokenizer,
        split="validation"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize encoders
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    
    # Initialize multimodal model
    model = MultimodalHyperbolicModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        projection_dim=embedding_dim,
        curvature=curvature
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting training")
    
    best_recall = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            text_embeddings, image_embeddings = model(
                text_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                image_inputs=pixel_values
            )
            
            # Compute loss
            loss = hyperbolic_contrastive_loss(
                z_text=text_embeddings,
                z_img=image_embeddings,
                temp=temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        text_embeddings_all = []
        image_embeddings_all = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                text_embeddings, image_embeddings = model(
                    text_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                    image_inputs=pixel_values
                )
                
                # Compute loss
                loss = hyperbolic_contrastive_loss(
                    z_text=text_embeddings,
                    z_img=image_embeddings,
                    temp=temperature
                )
                
                val_loss += loss.item()
                
                # Store embeddings for recall computation
                text_embeddings_all.append(text_embeddings)
                image_embeddings_all.append(image_embeddings)
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {avg_val_loss:.4f}")
        
        # Compute recall on a subset for efficiency
        text_embeddings_cat = torch.cat(text_embeddings_all[:10], dim=0)
        image_embeddings_cat = torch.cat(image_embeddings_all[:10], dim=0)
        
        recall_results = compute_recall_at_k(
            text_embeddings=text_embeddings_cat,
            image_embeddings=image_embeddings_cat,
            k_values=[1, 5, 10]
        )
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Recall results: {recall_results}")
        
        # Save checkpoint if improved
        current_recall = recall_results["r@1_text2image"]
        if current_recall > best_recall:
            best_recall = current_recall
            best_epoch = epoch + 1
            
            # Save model
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            
            # Save results
            with open(f"{output_dir}/best_results.json", "w") as f:
                json.dump({
                    "epoch": best_epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "recall": recall_results
                }, f, indent=4)
    
    logger.info(f"Training completed. Best model at epoch {best_epoch} with R@1: {best_recall:.4f}")
    
    # Load best model and evaluate on the full validation set
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
    model.eval()
    
    text_embeddings_all = []
    image_embeddings_all = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final evaluation"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            text_embeddings, image_embeddings = model(
                text_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                image_inputs=pixel_values
            )
            
            # Store embeddings for recall computation
            text_embeddings_all.append(text_embeddings)
            image_embeddings_all.append(image_embeddings)
    
    # Compute recall on the full validation set
    text_embeddings_cat = torch.cat(text_embeddings_all, dim=0)
    image_embeddings_cat = torch.cat(image_embeddings_all, dim=0)
    
    final_recall_results = compute_recall_at_k(
        text_embeddings=text_embeddings_cat,
        image_embeddings=image_embeddings_cat,
        k_values=[1, 5, 10]
    )
    
    logger.info(f"Final recall results: {final_recall_results}")
    
    # Save final results
    with open(f"{output_dir}/final_results.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_recall": best_recall,
            "final_recall": final_recall_results
        }, f, indent=4)
    
    return final_recall_results


def main(
    output_dir: str = "results/hyperbolic/v50000/retrieval",
    embedding_dim: int = 64,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 2e-5,
    temperature: float = 0.07,
    curvature: float = 1.0,
    seed: int = 42
) -> None:
    """
    Train a cross-modal retrieval model.
    """
    train_retrieval(
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        curvature=curvature,
        seed=seed
    )


if __name__ == "__main__":
    typer.run(main)
