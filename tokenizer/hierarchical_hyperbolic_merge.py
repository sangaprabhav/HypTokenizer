#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hierarchical Hyperbolic Tokenizer implementation.

This module implements a hyperbolic tokenizer with a staged, hierarchical
merge strategy that respects linguistic structure, progressively building from
character-level to word-level tokens.
"""

import torch
import numpy as np
import logging
import os
import sys
import re
from typing import List, Dict, Tuple, Optional, Set, Any
from tqdm import tqdm
import time
from collections import Counter

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import distance

# Try to import nltk for morphological analysis
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Morphological filtering will be limited.")

logger = logging.getLogger(__name__)


class HierarchicalHyperbolicTokenizer(HyperbolicTokenizer):
    """
    Hyperbolic tokenizer with hierarchical merge strategy.
    
    This tokenizer performs merges in distinct phases that respect linguistic structure:
    1. Character-level merges to build basic subwords
    2. Subword-level merges to build morphemes
    3. Word-level merges to build common words and compounds
    """
    
    def __init__(
        self, 
        vocab: List[str], 
        embeddings: torch.nn.Parameter,
        corpus_path: Optional[str] = None,
        curvature: float = 1.0,
        merge_threshold: float = 0.05,  # Start with a tighter threshold for character merges
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        max_vocab_size: int = 100000,
        use_approximate_search: bool = True,
        language: str = "english"
    ):
        """
        Initialize the hierarchical hyperbolic tokenizer.
        
        Args:
            vocab: Initial vocabulary (typically characters or character n-grams)
            embeddings: Initial embeddings in Lorentz model, shape (len(vocab), d+1)
            corpus_path: Path to corpus file for computing statistics
            curvature: Curvature parameter of the hyperbolic space
            merge_threshold: Initial threshold for considering merge candidates
            lr: Learning rate for embedding updates
            device: Device to use for computation
            max_vocab_size: Maximum vocabulary size
            use_approximate_search: Whether to use approximate search for large vocabularies
            language: Language for morphological analysis
        """
        super().__init__(
            vocab=vocab,
            embeddings=embeddings,
            curvature=curvature,
            merge_threshold=merge_threshold,
            lr=lr,
            device=device,
            max_vocab_size=max_vocab_size,
            use_approximate_search=use_approximate_search
        )
        
        # Language for morphological analysis
        self.language = language
        
        # Token statistics from corpus
        self.token_frequencies: Dict[str, int] = {}
        self.common_morphemes: Set[str] = set()
        self.common_words: Set[str] = set()
        
        # Initialize NLTK resources if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('wordnet', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK resources: {e}")
                
        # Compute statistics if corpus path is provided
        if corpus_path:
            self._compute_corpus_statistics(corpus_path)
    
    def _compute_corpus_statistics(self, corpus_path: str) -> None:
        """
        Compute corpus statistics for guiding hierarchical merges.
        
        Args:
            corpus_path: Path to corpus file
        """
        logger.info("Computing corpus statistics for hierarchical merging...")
        word_counter = Counter()
        subword_counter = Counter()
        
        # Process the corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing corpus"):
                # Tokenize into words first
                words = re.findall(r'\b\w+\b', line.lower())
                
                # Count words
                word_counter.update(words)
                
                # Extract potential morphemes (substrings)
                for word in words:
                    # Count character n-grams as potential subwords
                    for n in range(2, min(6, len(word) + 1)):
                        for i in range(len(word) - n + 1):
                            subword = word[i:i+n]
                            subword_counter[subword] += 1
        
        # Store token frequencies
        self.token_frequencies = dict(word_counter)
        
        # Identify common morphemes (frequent subwords)
        subword_threshold = np.percentile(list(subword_counter.values()), 80)
        self.common_morphemes = {
            subword for subword, count in subword_counter.items()
            if count >= subword_threshold
        }
        
        # Identify common words
        word_threshold = np.percentile(list(word_counter.values()), 70)
        self.common_words = {
            word for word, count in word_counter.items()
            if count >= word_threshold
        }
        
        logger.info(f"Identified {len(self.common_morphemes)} common morphemes and "
                   f"{len(self.common_words)} common words")
    
    def _is_potential_morpheme(self, token: str) -> bool:
        """
        Check if a token is a potential morpheme.
        
        Uses a combination of corpus statistics and linguistic rules.
        
        Args:
            token: Token to check
            
        Returns:
            Whether the token is a potential morpheme
        """
        # Check if it's in our identified common morphemes
        if token in self.common_morphemes:
            return True
        
        # Check based on morphological rules
        if NLTK_AVAILABLE:
            # Common prefixes in English
            common_prefixes = {'re', 'un', 'in', 'im', 'il', 'ir', 'dis', 'en', 'em', 'non', 'de', 'pre', 'pro', 'mis'}
            # Common suffixes in English
            common_suffixes = {'ing', 'ed', 'er', 'est', 'ly', 'ity', 'ment', 'ness', 'able', 'ible', 'al', 'ial'}
            
            # Check if token is a common prefix or suffix
            if token in common_prefixes or token in common_suffixes:
                return True
            
            # Try to find in WordNet
            if len(token) > 2:  # Minimum length for meaningful morpheme
                # Check if any word with this morpheme exists in WordNet
                for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
                    if wordnet.synsets(token, pos=pos):
                        return True
        
        # Fallback heuristics
        if len(token) >= 2 and len(token) <= 5:
            # If the token appears frequently as a substring in the corpus
            frequency = sum(1 for word in self.common_words if token in word)
            if frequency >= 5:  # Arbitrary threshold
                return True
        
        return False
    
    def _is_valid_word(self, token: str) -> bool:
        """
        Check if a token is a valid word.
        
        Args:
            token: Token to check
            
        Returns:
            Whether the token is a valid word
        """
        # Check if it's in our common words
        if token in self.common_words:
            return True
        
        # Check if it's a dictionary word using NLTK
        if NLTK_AVAILABLE:
            # Check if it exists in WordNet
            if wordnet.synsets(token):
                return True
        
        # Fallback: check if it's long enough and has vowels (basic heuristic)
        if len(token) >= 3 and re.search(r'[aeiou]', token):
            return True
            
        return False
    
    def _filter_morphologically_valid(self, candidates: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Filter merge candidates to prefer morphologically valid merges.
        
        Args:
            candidates: List of (i, j, distance) merge candidates
            
        Returns:
            Filtered list of candidates
        """
        valid_candidates = []
        
        for i, j, dist in candidates:
            token_i, token_j = self.vocab[i], self.vocab[j]
            merged = token_i + token_j
            
            # Check if the merged token forms a potentially valid morpheme
            if self._is_potential_morpheme(merged):
                # Give higher priority to morphologically valid merges
                valid_candidates.append((i, j, dist * 0.8))  # Lower distance = higher priority
            else:
                # Keep other candidates but with lower priority
                valid_candidates.append((i, j, dist))
        
        return valid_candidates
    
    def _filter_word_valid(self, candidates: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Filter merge candidates to prefer valid word merges.
        
        Args:
            candidates: List of (i, j, distance) merge candidates
            
        Returns:
            Filtered list of candidates
        """
        valid_candidates = []
        
        for i, j, dist in candidates:
            token_i, token_j = self.vocab[i], self.vocab[j]
            merged = token_i + token_j
            
            # Check if the merged token forms a valid word
            if self._is_valid_word(merged):
                # Give higher priority to valid word merges
                valid_candidates.append((i, j, dist * 0.7))  # Lower distance = higher priority
            else:
                # Keep other candidates but with lower priority
                valid_candidates.append((i, j, dist))
        
        return valid_candidates
    
    def _hierarchical_merge_strategy(self, target_vocab_size: Optional[int] = None) -> None:
        """
        Perform merges in a hierarchical manner respecting linguistic structure.
        
        Args:
            target_vocab_size: Target vocabulary size (if None, follows the phases)
        """
        from tqdm import tqdm
        
        current_size = self.current_vocab_size
        logger.info(f"Starting hierarchical merge with vocabulary size: {current_size}")
        
        # Phase 1: Character-level merges (build basic subwords)
        self.merge_threshold = 0.05
        logger.info("Phase 1: Character-level merges (building basic subwords)")
        
        pbar = tqdm(range(2000), desc="Phase 1: Character merges")
        phase1_count = 0
        
        for step in pbar:
            candidates = self._find_merge_candidates()
            if not candidates:
                # Gradually increase threshold if needed
                if phase1_count < 500:  # Aim for at least 500 merges in phase 1
                    self.merge_threshold *= 1.2
                    logger.info(f"Increasing threshold to {self.merge_threshold:.4f}")
                    continue
                else:
                    break
            
            # Filter for character-level merges in phase 1
            char_candidates = [c for c in candidates if len(self.vocab[c[0]]) <= 2 and len(self.vocab[c[1]]) <= 2]
            
            if not char_candidates and phase1_count < 500:
                # If no character-level candidates but we want more, relax the criteria
                char_candidates = [c for c in candidates if len(self.vocab[c[0]]) <= 3 and len(self.vocab[c[1]]) <= 3]
            
            if char_candidates:
                best = min(char_candidates, key=lambda x: x[2])
                self._merge_tokens(best[0], best[1])
                phase1_count += 1
                
                # Log progress
                if step % 100 == 0:
                    i, j = best[0], best[1]
                    logger.info(f"Merged '{self.vocab[i]}' + '{self.vocab[j]}' → '{self.vocab[-1]}'")
            else:
                # If no character candidates left, move to phase 2
                break
            
            # Check if target size reached
            if target_vocab_size and self.current_vocab_size >= target_vocab_size:
                logger.info(f"Reached target vocabulary size {target_vocab_size}")
                return
            
            pbar.set_postfix({"vocab_size": self.current_vocab_size, "phase1_merges": phase1_count})
        
        logger.info(f"Completed Phase 1 with {phase1_count} merges. Vocabulary size: {self.current_vocab_size}")
        
        # Phase 2: Subword-level merges (build morphemes)
        self.merge_threshold = 0.1
        logger.info("Phase 2: Subword-level merges (building morphemes)")
        
        pbar = tqdm(range(5000), desc="Phase 2: Subword merges")
        phase2_count = 0
        
        for step in pbar:
            candidates = self._find_merge_candidates()
            if not candidates:
                # Gradually increase threshold if needed
                if phase2_count < 2000:  # Aim for significant subword merges
                    self.merge_threshold *= 1.2
                    logger.info(f"Increasing threshold to {self.merge_threshold:.4f}")
                    continue
                else:
                    break
            
            # Filter for morphologically valid merges
            filtered_candidates = self._filter_morphologically_valid(candidates)
            
            if filtered_candidates:
                best = min(filtered_candidates, key=lambda x: x[2])
                self._merge_tokens(best[0], best[1])
                phase2_count += 1
                
                # Log progress
                if step % 100 == 0:
                    i, j = best[0], best[1]
                    logger.info(f"Merged '{self.vocab[i]}' + '{self.vocab[j]}' → '{self.vocab[-1]}'")
            else:
                # If no suitable candidates, move to next phase
                break
            
            # Check if target size reached
            if target_vocab_size and self.current_vocab_size >= target_vocab_size:
                logger.info(f"Reached target vocabulary size {target_vocab_size}")
                return
            
            pbar.set_postfix({"vocab_size": self.current_vocab_size, "phase2_merges": phase2_count})
        
        logger.info(f"Completed Phase 2 with {phase2_count} merges. Vocabulary size: {self.current_vocab_size}")
        
        # Phase 3: Word-level merges (build compounds and common words)
        self.merge_threshold = 0.2
        logger.info("Phase 3: Word-level merges (building words and compounds)")
        
        pbar = tqdm(range(10000), desc="Phase 3: Word merges")
        phase3_count = 0
        
        for step in pbar:
            candidates = self._find_merge_candidates()
            if not candidates:
                # Gradually increase threshold if needed
                if phase3_count < 5000 and self.merge_threshold < 1.0:
                    self.merge_threshold *= 1.2
                    logger.info(f"Increasing threshold to {self.merge_threshold:.4f}")
                    continue
                else:
                    break
            
            # Filter for word-valid merges
            filtered_candidates = self._filter_word_valid(candidates)
            
            if filtered_candidates:
                best = min(filtered_candidates, key=lambda x: x[2])
                self._merge_tokens(best[0], best[1])
                phase3_count += 1
                
                # Log progress
                if step % 100 == 0:
                    i, j = best[0], best[1]
                    logger.info(f"Merged '{self.vocab[i]}' + '{self.vocab[j]}' → '{self.vocab[-1]}'")
            else:
                # If no suitable candidates, continue with regular merges
                if candidates:
                    best = min(candidates, key=lambda x: x[2])
                    self._merge_tokens(best[0], best[1])
                    phase3_count += 1
                else:
                    break
            
            # Check if target size reached
            if target_vocab_size and self.current_vocab_size >= target_vocab_size:
                logger.info(f"Reached target vocabulary size {target_vocab_size}")
                return
            
            pbar.set_postfix({"vocab_size": self.current_vocab_size, "phase3_merges": phase3_count})
        
        logger.info(f"Completed Phase 3 with {phase3_count} merges. Vocabulary size: {self.current_vocab_size}")
        logger.info(f"Final vocabulary size: {self.current_vocab_size}")
    
    def optimize_merges(self, 
                        steps: int = 10000, 
                        log_every: int = 1000,
                        hierarchical: bool = True,
                        target_vocab_size: Optional[int] = None) -> None:
        """
        Perform merge optimization to build the vocabulary.
        
        Args:
            steps: Maximum number of merge steps to perform (used only if hierarchical=False)
            log_every: How often to log progress
            hierarchical: Whether to use the hierarchical merge strategy
            target_vocab_size: Target vocabulary size
        """
        if hierarchical:
            self._hierarchical_merge_strategy(target_vocab_size)
        else:
            # Fall back to regular merge strategy
            super().optimize_merges(steps, log_every)
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            path: Directory path to save the tokenizer
        """
        # Call parent implementation first
        super().save(path)
        
        # Save hierarchical tokenizer specific data
        import json
        
        # Save morpheme and word data
        hierarchical_data = {
            "language": self.language,
            "common_morphemes": list(self.common_morphemes),
            "common_words": list(self.common_words)
        }
        
        with open(f"{path}/hierarchical_data.json", "w") as f:
            json.dump(hierarchical_data, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'HierarchicalHyperbolicTokenizer':
        """
        Load a tokenizer from disk.
        
        Args:
            path: Directory path to load the tokenizer from
            device: Device to load the tokenizer onto
            
        Returns:
            Loaded tokenizer
        """
        import json
        
        # Load base tokenizer first
        tokenizer = super().load(path, device)
        
        # Convert to HierarchicalHyperbolicTokenizer
        hierarchical_tokenizer = cls(
            vocab=tokenizer.vocab,
            embeddings=tokenizer.embeddings,
            curvature=tokenizer.curvature,
            merge_threshold=tokenizer.merge_threshold,
            device=device if device is not None else tokenizer.device
        )
        
        # Copy over attributes
        hierarchical_tokenizer.merge_history = tokenizer.merge_history
        hierarchical_tokenizer.current_vocab_size = tokenizer.current_vocab_size
        
        # Load hierarchical data if available
        try:
            with open(f"{path}/hierarchical_data.json", "r") as f:
                hierarchical_data = json.load(f)
                hierarchical_tokenizer.language = hierarchical_data.get("language", "english")
                hierarchical_tokenizer.common_morphemes = set(hierarchical_data.get("common_morphemes", []))
                hierarchical_tokenizer.common_words = set(hierarchical_data.get("common_words", []))
        except FileNotFoundError:
            logger.warning("Hierarchical data file not found")
        
        return hierarchical_tokenizer
