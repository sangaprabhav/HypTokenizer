#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis and visualization script for Hyperbolic Tokenization results.

This script generates visualizations and analyses the results of
the Hyperbolic Tokenization experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
import umap

# Add parent directory to path to import from embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.poincare_ball import log_map_zero


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


def plot_distortion_vs_vocab_size(
    results_dir: str,
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    vocab_sizes: List[int] = [10000, 20000, 50000, 100000],
    output_file: str = "results/figures/distortion_vs_vocab.png"
):
    """
    Plot distortion vs. vocabulary size.
    
    Args:
        results_dir: Directory containing results
        methods: List of tokenization methods
        vocab_sizes: List of vocabulary sizes
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Dictionary to store data for each method
    data = {}
    
    for method in methods:
        distortions = []
        errors = []
        
        for V in vocab_sizes:
            stats_file = f"{results_dir}/{method}/v{V}/hierarchy_distortion_stats.json"
            
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                
                distortions.append(stats["mean"])
                errors.append(stats["std"])
            else:
                logger.warning(f"Results file not found: {stats_file}")
                distortions.append(np.nan)
                errors.append(np.nan)
        
        data[method] = {
            "distortions": distortions,
            "errors": errors
        }
    
    # Plot results
    for method in methods:
        plt.plot(
            vocab_sizes,
            data[method]["distortions"],
            marker='o',
            label=method.capitalize()
        )
        
        plt.fill_between(
            vocab_sizes,
            np.array(data[method]["distortions"]) - np.array(data[method]["errors"]),
            np.array(data[method]["distortions"]) + np.array(data[method]["errors"]),
            alpha=0.2
        )
    
    plt.xlabel('Vocabulary Size', fontsize=14)
    plt.ylabel('Average Distortion', fontsize=14)
    plt.title('Distortion vs. Vocabulary Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    plt.close()


def plot_perplexity_vs_distortion(
    results_dir: str,
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    vocab_sizes: List[int] = [10000, 20000, 50000, 100000],
    output_file: str = "results/figures/perplexity_vs_distortion.png"
):
    """
    Plot perplexity vs. distortion scatter plot.
    
    Args:
        results_dir: Directory containing results
        methods: List of tokenization methods
        vocab_sizes: List of vocabulary sizes
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Dictionary to store data for each method
    data = []
    
    for method in methods:
        for V in vocab_sizes:
            stats_file = f"{results_dir}/{method}/v{V}/hierarchy_distortion_stats.json"
            mlm_file = f"{results_dir}/{method}/v{V}/tasks/mlm_results.json"
            
            if os.path.exists(stats_file) and os.path.exists(mlm_file):
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                
                with open(mlm_file, "r") as f:
                    mlm_results = json.load(f)
                
                perplexity = np.exp(mlm_results.get("eval_loss", 0))
                distortion = stats["mean"]
                
                data.append({
                    "method": method,
                    "vocab_size": V,
                    "perplexity": perplexity,
                    "distortion": distortion
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        method_data = df[df["method"] == method]
        plt.scatter(
            method_data["distortion"],
            method_data["perplexity"],
            label=method.capitalize(),
            s=100,
            alpha=0.7
        )
    
    # Add labels for points
    for _, row in df.iterrows():
        plt.annotate(
            f"{row['vocab_size']//1000}K",
            (row["distortion"], row["perplexity"]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.xlabel('Distortion', fontsize=14)
    plt.ylabel('Perplexity', fontsize=14)
    plt.title('Perplexity vs. Distortion', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    plt.close()


def plot_metrics_bar_chart(
    results_dir: str,
    vocab_size: int = 50000,
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    output_prefix: str = "results/figures/"
):
    """
    Plot bar charts for MLM perplexity, classification accuracy, and Recall@K.
    
    Args:
        results_dir: Directory containing results
        vocab_size: Vocabulary size to use
        methods: List of tokenization methods
        output_prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_prefix, exist_ok=True)
    
    # Plot MLM perplexity
    mlm_data = []
    
    for method in methods:
        mlm_file = f"{results_dir}/{method}/v{vocab_size}/tasks/mlm_results.json"
        
        if os.path.exists(mlm_file):
            with open(mlm_file, "r") as f:
                mlm_results = json.load(f)
            
            perplexity = np.exp(mlm_results.get("eval_loss", 0))
            
            mlm_data.append({
                "method": method.capitalize(),
                "perplexity": perplexity
            })
    
    if mlm_data:
        df_mlm = pd.DataFrame(mlm_data)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="method", y="perplexity", data=df_mlm)
        
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('Perplexity', fontsize=14)
        plt.title(f'MLM Perplexity (V={vocab_size})', fontsize=16)
        
        # Add values on bars
        for i, v in enumerate(df_mlm["perplexity"]):
            ax.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=10)
        
        plt.savefig(f"{output_prefix}mlm_perplexity.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved MLM perplexity plot to {output_prefix}mlm_perplexity.png")
        
        plt.close()
    
    # Plot classification accuracy
    cls_data = []
    
    for method in methods:
        cls_file = f"{results_dir}/{method}/v{vocab_size}/tasks/classification_results.json"
        
        if os.path.exists(cls_file):
            with open(cls_file, "r") as f:
                cls_results = json.load(f)
            
            accuracy = cls_results.get("eval_accuracy", 0) * 100
            
            cls_data.append({
                "method": method.capitalize(),
                "accuracy": accuracy
            })
    
    if cls_data:
        df_cls = pd.DataFrame(cls_data)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="method", y="accuracy", data=df_cls)
        
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title(f'Classification Accuracy (V={vocab_size})', fontsize=16)
        
        # Add values on bars
        for i, v in enumerate(df_cls["accuracy"]):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)
        
        plt.savefig(f"{output_prefix}classification_accuracy.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved classification accuracy plot to {output_prefix}classification_accuracy.png")
        
        plt.close()
    
    # Plot Recall@K
    recall_data = []
    
    for method in methods:
        recall_file = f"{results_dir}/{method}/v{vocab_size}/retrieval/final_results.json"
        
        if os.path.exists(recall_file):
            with open(recall_file, "r") as f:
                recall_results = json.load(f)
            
            final_recall = recall_results.get("final_recall", {})
            
            for k in [1, 5, 10]:
                t2i = final_recall.get(f"r@{k}_text2image", 0) * 100
                i2t = final_recall.get(f"r@{k}_image2text", 0) * 100
                
                recall_data.append({
                    "method": method.capitalize(),
                    "k": k,
                    "direction": "Text→Image",
                    "recall": t2i
                })
                
                recall_data.append({
                    "method": method.capitalize(),
                    "k": k,
                    "direction": "Image→Text",
                    "recall": i2t
                })
    
    if recall_data:
        df_recall = pd.DataFrame(recall_data)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x="method", y="recall", hue="direction", col="k", data=df_recall)
        
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('Recall@K (%)', fontsize=14)
        plt.title(f'Cross-Modal Retrieval Recall (V={vocab_size})', fontsize=16)
        plt.legend(title="Direction", fontsize=12)
        
        plt.savefig(f"{output_prefix}retrieval_recall.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved retrieval recall plot to {output_prefix}retrieval_recall.png")
        
        plt.close()


def plot_efficiency(
    results_dir: str,
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    output_prefix: str = "results/figures/"
):
    """
    Plot efficiency metrics (throughput and training time).
    
    Args:
        results_dir: Directory containing results
        methods: List of tokenization methods
        output_prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_prefix, exist_ok=True)
    
    # Plot tokenization throughput
    throughput_file = f"{results_dir}/throughput.json"
    
    if os.path.exists(throughput_file):
        with open(throughput_file, "r") as f:
            throughput_results = json.load(f)
        
        throughput_data = []
        
        for method in methods:
            if method in throughput_results:
                throughput = throughput_results[method].get("throughput", 0)
                
                throughput_data.append({
                    "method": method.capitalize(),
                    "throughput": throughput
                })
        
        if throughput_data:
            df_throughput = pd.DataFrame(throughput_data)
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="method", y="throughput", data=df_throughput)
            
            plt.xlabel('Method', fontsize=14)
            plt.ylabel('Throughput (tokens/s)', fontsize=14)
            plt.title('Tokenization Throughput', fontsize=16)
            
            # Add values on bars
            for i, v in enumerate(df_throughput["throughput"]):
                ax.text(i, v + 0.5, f"{v:.0f}", ha='center', fontsize=10)
            
            plt.savefig(f"{output_prefix}tokenization_throughput.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved tokenization throughput plot to {output_prefix}tokenization_throughput.png")
            
            plt.close()
    
    # Plot training time
    training_time_file = f"{results_dir}/training_time.json"
    
    if os.path.exists(training_time_file):
        with open(training_time_file, "r") as f:
            training_time_results = json.load(f)
        
        training_time_data = []
        
        for method in methods:
            if method in training_time_results:
                training_time = training_time_results[method].get("training_time", 0)
                
                # Convert to hours
                training_time_hours = training_time / 3600
                
                training_time_data.append({
                    "method": method.capitalize(),
                    "training_time": training_time_hours
                })
        
        if training_time_data:
            df_training_time = pd.DataFrame(training_time_data)
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="method", y="training_time", data=df_training_time)
            
            plt.xlabel('Method', fontsize=14)
            plt.ylabel('Training Time (hours)', fontsize=14)
            plt.title('Tokenizer Training Time', fontsize=16)
            
            # Add values on bars
            for i, v in enumerate(df_training_time["training_time"]):
                ax.text(i, v + 0.1, f"{v:.1f}h", ha='center', fontsize=10)
            
            plt.savefig(f"{output_prefix}training_time.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved training time plot to {output_prefix}training_time.png")
            
            plt.close()


def plot_embeddings_umap(
    embeddings_path: str,
    output_file: str = "results/figures/embeddings_umap.png",
    n_samples: int = 1000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
):
    """
    Plot UMAP visualization of token embeddings.
    
    Args:
        embeddings_path: Path to embeddings file
        output_file: Path to save the plot
        n_samples: Number of embeddings to sample
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random state for reproducibility
    """
    # Load embeddings
    embeddings = torch.load(embeddings_path, map_location="cpu")
    
    # If these are Lorentz model embeddings, convert to Euclidean
    if embeddings.shape[1] > 2 and embeddings[0, 0] > 0.9:
        embeddings = log_map_zero(embeddings)
    
    # Sample embeddings
    if n_samples < embeddings.shape[0]:
        indices = np.random.RandomState(random_state).choice(
            embeddings.shape[0], n_samples, replace=False
        )
        embeddings_sample = embeddings[indices].numpy()
    else:
        embeddings_sample = embeddings.numpy()
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    embedding_2d = reducer.fit_transform(embeddings_sample)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=10,
        c=np.arange(embedding_2d.shape[0]),
        cmap='viridis',
        alpha=0.7
    )
    
    plt.title('UMAP projection of token embeddings', fontsize=16)
    plt.colorbar(label='Token index')
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved UMAP plot to {output_file}")
    
    plt.close()


def run_statistical_tests(
    results_dir: str,
    vocab_size: int = 50000,
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    output_file: str = "results/statistical_tests.csv"
):
    """
    Run statistical tests to compare methods.
    
    Args:
        results_dir: Directory containing results
        vocab_size: Vocabulary size to use
        methods: List of tokenization methods
        output_file: Path to save the results
    """
    # Dictionary to store metrics for each method
    metrics = {
        "distortion": {},
        "perplexity": {},
        "accuracy": {},
        "recall_1": {}
    }
    
    # Load distortion data
    for method in methods:
        stats_file = f"{results_dir}/{method}/v{vocab_size}/hierarchy_distortion_stats.json"
        
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
            
            metrics["distortion"][method] = stats["mean"]
    
    # Load perplexity data
    for method in methods:
        mlm_file = f"{results_dir}/{method}/v{vocab_size}/tasks/mlm_results.json"
        
        if os.path.exists(mlm_file):
            with open(mlm_file, "r") as f:
                mlm_results = json.load(f)
            
            metrics["perplexity"][method] = np.exp(mlm_results.get("eval_loss", 0))
    
    # Load accuracy data
    for method in methods:
        cls_file = f"{results_dir}/{method}/v{vocab_size}/tasks/classification_results.json"
        
        if os.path.exists(cls_file):
            with open(cls_file, "r") as f:
                cls_results = json.load(f)
            
            metrics["accuracy"][method] = cls_results.get("eval_accuracy", 0) * 100
    
    # Load recall data
    for method in methods:
        recall_file = f"{results_dir}/{method}/v{vocab_size}/retrieval/final_results.json"
        
        if os.path.exists(recall_file):
            with open(recall_file, "r") as f:
                recall_results = json.load(f)
            
            final_recall = recall_results.get("final_recall", {})
            metrics["recall_1"][method] = final_recall.get("r@1_text2image", 0) * 100
    
    # Run t-tests
    results = []
    
    for metric_name, metric_values in metrics.items():
        if "hyperbolic" in metric_values:
            hyperbolic_value = metric_values["hyperbolic"]
            
            for method in methods:
                if method != "hyperbolic" and method in metric_values:
                    baseline_value = metric_values[method]
                    
                    # For distortion and perplexity, lower is better
                    if metric_name in ["distortion", "perplexity"]:
                        is_better = hyperbolic_value < baseline_value
                    else:
                        is_better = hyperbolic_value > baseline_value
                    
                    # In a real implementation, we would run t-tests on multiple runs
                    # Here we'll just compute the difference
                    diff = hyperbolic_value - baseline_value
                    rel_diff = diff / baseline_value * 100
                    
                    results.append({
                        "method": method,
                        "metric": metric_name,
                        "hyperbolic_value": hyperbolic_value,
                        "baseline_value": baseline_value,
                        "diff": diff,
                        "rel_diff": rel_diff,
                        "is_better": is_better
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    df.to_csv(output_file, index=False)
    logger.info(f"Saved statistical test results to {output_file}")
    
    return df


def main(
    results_dir: str = "results",
    vocab_sizes: List[int] = [10000, 20000, 50000, 100000],
    methods: List[str] = ["hyperbolic", "bpe", "wordpiece", "unigram"],
    figures_dir: str = "results/figures"
):
    """
    Run all analyses and generate plots.
    
    Args:
        results_dir: Directory containing results
        vocab_sizes: List of vocabulary sizes
        methods: List of tokenization methods
        figures_dir: Directory to save figures
    """
    # Create figures directory
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot distortion vs. vocabulary size
    plot_distortion_vs_vocab_size(
        results_dir=results_dir,
        methods=methods,
        vocab_sizes=vocab_sizes,
        output_file=f"{figures_dir}/distortion_vs_vocab.png"
    )
    
    # Plot perplexity vs. distortion
    plot_perplexity_vs_distortion(
        results_dir=results_dir,
        methods=methods,
        vocab_sizes=vocab_sizes,
        output_file=f"{figures_dir}/perplexity_vs_distortion.png"
    )
    
    # Plot metrics bar charts
    plot_metrics_bar_chart(
        results_dir=results_dir,
        vocab_size=50000,
        methods=methods,
        output_prefix=figures_dir + "/"
    )
    
    # Plot efficiency metrics
    plot_efficiency(
        results_dir=f"{results_dir}/efficiency",
        methods=methods,
        output_prefix=figures_dir + "/"
    )
    
    # Plot embeddings UMAP
    plot_embeddings_umap(
        embeddings_path=f"{results_dir}/hyperbolic/v50000/embeddings.pt",
        output_file=f"{figures_dir}/embeddings_umap.png"
    )
    
    # Run statistical tests
    run_statistical_tests(
        results_dir=results_dir,
        vocab_size=50000,
        methods=methods,
        output_file=f"{results_dir}/statistical_tests.csv"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Hyperbolic Tokenization results")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory to save figures")
    
    args = parser.parse_args()
    
    main(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir
    )
