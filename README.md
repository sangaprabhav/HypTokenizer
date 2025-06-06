![Project Banner](ss.png)

# Hyperbolic Tokenization Framework

This repository contains a complete implementation of the Hyperbolic Tokenization framework—a novel approach to subword tokenization that leverages hyperbolic geometry to guide the token merging process. The implementation includes both the standard algorithm and an optimized version with Hierarchical Navigable Small World (HNSW) indexing for efficient nearest neighbor search in large vocabularies.

## 📋 Overview

Traditional subword tokenization methods like BPE, WordPiece, and Unigram rely on frequency-based or likelihood-based merge strategies in Euclidean space. The Hyperbolic Tokenization framework introduces a fundamentally different approach by:

1. Embedding tokens in hyperbolic space (Lorentz model)
2. Using hyperbolic distance to guide merge decisions
3. Preserving hierarchical relationships in the vocabulary
4. Enabling improved downstream task performance
5. Efficient scaling to large vocabularies with optimized algorithms

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA (optional, for GPU acceleration)
- FAISS (optional, for efficient nearest neighbor search)
- Apple MPS (optional, for acceleration on Apple Silicon)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HypTokenizer.git
cd HypTokenizer

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Project Structure

```
HypTokenizer/
├── data/                       # Data storage
│   ├── raw/                    # Raw downloaded datasets
│   └── processed/              # Processed datasets
├── tokenizer/                  # Tokenization algorithms
│   ├── hyperbolic_merge.py     # Hyperbolic tokenizer implementation
│   ├── fast_hyperbolic_merge.py # Optimized tokenizer with HNSW
│   ├── frequency_aware_hyperbolic_merge.py # Frequency-aware tokenizer
│   ├── hierarchical_hyperbolic_merge.py # Hierarchical merge strategy tokenizer
│   ├── adaptive_curvature_tokenizer.py # Adaptive curvature optimization tokenizer
│   ├── compression_aware_tokenizer.py # Compression-aware tokenizer
│   └── enhanced_fast_hyperbolic_merge.py # Enhanced tokenizer with all features
├── embedding/              # Hyperbolic embedding modules
│   ├── lorentz_model.py    # Lorentz model operations
│   └── poincare_ball.py    # Poincaré ball operations
├── multimodal/             # Multimodal learning components
│   └── contrastive_loss.py # Hyperbolic contrastive loss
├── scripts/                # Training and evaluation scripts
├── experiments/            # Experiment configurations
├── notebooks/              # Analysis notebooks
├── results/                # Experimental results
└── tests/                  # Unit tests
```

## 🛠️ Usage

### Data Preprocessing

```bash
# Build WordNet graph
python scripts/build_wordnet_graph.py

# Preprocess Wikipedia data
python scripts/preprocess_wiki.py --input_dir data/raw/wiki --output_file data/processed/wiki/wiki.txt
```

### Training Tokenizers

```bash
# Train baseline tokenizers (BPE, WordPiece, Unigram)
# For each method ∈ {bpe, wordpiece, unigram} and each V ∈ {10000, 20000, 50000, 100000}

# Train standard hyperbolic tokenizer
python scripts/train_hyperbolic_tokenizer.py \
    --vocab_path data/processed/wiki/vocab_initial.txt \
    --output_dir results/hyperbolic/v50000 \
    --embedding_dim 50 \
    --target_vocab_size 50000

# Train optimized hyperbolic tokenizer with HNSW
python scripts/train_hyperbolic_tokenizer.py \
    --vocab_path data/processed/wiki/vocab_initial.txt \
    --output_dir results/hyperbolic/fast_tokenizer \
    --embedding_dim 100 \
    --target_vocab_size 50000 \
    --use-fast-tokenizer \
    --hnsw_m 32 \
    --hnsw_ef_construction 200 \
    --hnsw_ef_search 100 \
    --cache_size 10000

# Train enhanced hyperbolic tokenizer with all advanced features
python scripts/train_enhanced_hyperbolic_tokenizer.py \
    --vocab_path data/processed/wiki/vocab_initial.txt \
    --corpus_path data/processed/wiki/wiki.txt \
    --output_dir results/hyperbolic/enhanced_tokenizer \
    --embedding_dim 100 \
    --target_vocab_size 50000 \
    --use_frequency_aware \
    --use_hierarchical \
    --use_adaptive_curvature \
    --use_compression_aware \
    --phase_transition_steps 1000 6000

# For Apple Silicon (MPS) or platforms without FAISS
python scripts/train_hyperbolic_tokenizer.py \
    --vocab_path data/processed/wiki/vocab_initial.txt \
    --output_dir results/hyperbolic/fast_tokenizer_no_faiss \
    --embedding_dim 100 \
    --target_vocab_size 50000 \
    --use-fast-tokenizer \
    --no-faiss
```

### Downstream Task Evaluation

```bash
# Train and evaluate on NLP tasks
python scripts/train_nlp_tasks.py \
    --method hyperbolic \
    --vocab_size 50000 \
    --model_path results/hyperbolic/v50000

# Train and evaluate on cross-modal retrieval
python scripts/train_retrieval.py \
    --output_dir results/hyperbolic/v50000/retrieval
```

### Evaluation and Analysis

```bash
# Evaluate hierarchy distortion
python scripts/eval_hierarchy.py \
    --embeddings_path results/hyperbolic/v50000/embeddings.pt \
    --vocab_path results/hyperbolic/v50000/vocab.json

# Benchmark efficiency
python scripts/benchmark_efficiency.py

# Generate visualizations and analyses
python notebooks/analysis.py
```

## 📝 Key Components

### Hyperbolic Embedding

The framework implements the Lorentz model of hyperbolic geometry, with operations including:
- Minkowski dot product and norm
- Exponential and logarithmic maps
- Parallel transport
- Riemannian gradient

### Hyperbolic Tokenizer

The framework includes several tokenizer implementations:

**Standard HyperbolicTokenizer:**
- Initialized with character-level tokens and embeddings
- Iteratively merges tokens based on hyperbolic distance
- Uses hyperbolic midpoint calculation for new token embeddings
- Implements standard tokenize/encode/decode interface

**FastHyperbolicTokenizer (Optimized):**
- Uses HNSW (Hierarchical Navigable Small World) indexing for efficient nearest neighbor search
- Implements batch-based distance calculations using Einstein summation for performance
- Provides adaptive caching strategy for merge candidates
- Supports parallel candidate evaluation for faster merges
- Includes optimizations for different device types (CUDA, MPS, CPU)
- Automatically falls back to optimized batch method when FAISS is unavailable

**EnhancedFastHyperbolicTokenizer (Advanced):**
- Integrates multiple advanced tokenization techniques in a unified framework
- **Frequency-Aware Merging**: Uses corpus statistics to prioritize frequent token pairs
- **Hierarchical Merge Strategy**: Three-phase approach (character → subword → word-level)
- **Adaptive Curvature Optimization**: Dynamically adjusts curvature to preserve hierarchy
- **Compression-Aware Scoring**: Evaluates merge candidates based on compression efficiency
- Configurable feature flags allow enabling/disabling individual components
- Provides phase-based training with adaptive thresholds for different linguistic levels
- Comprehensive diagnostics and statistics tracking during training

### Multimodal Learning

The framework includes components for multimodal learning in hyperbolic space:
- Hyperbolic contrastive loss
- Two-tower architecture for text-image retrieval
- Projection layers for mapping to hyperbolic space

## 📊 Results

After running the evaluation pipeline, you can analyze the results in `results/figures/`:
- Distortion vs. vocabulary size
- Perplexity vs. distortion
- Comparative metrics on downstream tasks
- Efficiency benchmarks

### Performance Improvements

The optimized FastHyperbolicTokenizer provides significant performance improvements:
- Up to 10-100x faster nearest neighbor search with HNSW indexing for large vocabularies (50k+)
- 2-5x faster merge operations with batch-based distance calculations
- Efficient memory usage with pre-allocated embeddings
- O(1) lookups for tokenization with merge rules dictionary
- Smart caching strategy to reduce redundant computations

## 🔬 Tests

Run the test suite to verify the implementation:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_lorentz_model
```

## 📚 References

- Dhingra, B., Faruqui, M., et al. (2018). Embedding Text in Hyperbolic Spaces.
- Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations.
- Ganea, O., Bécigneul, G., et al. (2018). Hyperbolic Neural Networks.
- Tifrea, A., Bécigneul, G., et al. (2018). Poincaré GloVe: Hyperbolic Word Embeddings.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
