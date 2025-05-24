#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the HyperbolicTokenizer class.
"""

import os
import sys
import unittest
import torch
import tempfile
import shutil

# Add parent directory to path to import from tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.hyperbolic_merge import HyperbolicTokenizer
from embedding.lorentz_model import distance, project_to_hyperboloid


class TestHyperbolicTokenizer(unittest.TestCase):
    """Test cases for the HyperbolicTokenizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a simple vocabulary
        self.vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c", "d", "e"]
        
        # Initialize embeddings in hyperbolic space
        self.embedding_dim = 5
        tangent = torch.randn((len(self.vocab), self.embedding_dim), dtype=torch.float32) * 0.01
        
        # Origin point in Lorentz model
        origin = torch.zeros(self.embedding_dim + 1, dtype=torch.float32)
        origin[0] = 1.0
        
        # Map to hyperboloid
        self.embeddings = torch.zeros((len(self.vocab), self.embedding_dim + 1), dtype=torch.float32)
        
        for i, t in enumerate(tangent):
            from embedding.lorentz_model import exp_map
            p = exp_map(origin.unsqueeze(0), 
                        torch.cat([torch.zeros(1), t]).unsqueeze(0))[0]
            self.embeddings[i] = p
        
        # Ensure all points are on the hyperboloid
        self.embeddings = project_to_hyperboloid(self.embeddings)
        
        # Create the tokenizer
        self.curvature = 1.0
        self.merge_threshold = 0.5
        self.lr = 1e-3
        
        self.tokenizer = HyperbolicTokenizer(
            vocab=self.vocab,
            embeddings=torch.nn.Parameter(self.embeddings),
            curvature=self.curvature,
            merge_threshold=self.merge_threshold,
            lr=self.lr
        )
        
        # Create a temporary directory for saving/loading
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        # Check that the vocabulary was set correctly
        self.assertEqual(self.tokenizer.vocab, self.vocab)
        
        # Check that the embeddings were set correctly
        self.assertTrue(torch.allclose(self.tokenizer.embeddings, self.embeddings))
        
        # Check that the token to index mapping was created correctly
        for i, token in enumerate(self.vocab):
            self.assertEqual(self.tokenizer.token2idx[token], i)
        
        # Check that the parameters were set correctly
        self.assertEqual(self.tokenizer.curvature, self.curvature)
        self.assertEqual(self.tokenizer.merge_threshold, self.merge_threshold)
        self.assertEqual(self.tokenizer.lr, self.lr)
    
    def test_compute_pairwise_distances(self):
        """Test computation of pairwise distances."""
        # Compute pairwise distances
        distances = self.tokenizer._compute_pairwise_distances()
        
        # Check shape
        self.assertEqual(distances.shape, (len(self.vocab), len(self.vocab)))
        
        # Check that the diagonal is 0
        for i in range(len(self.vocab)):
            self.assertAlmostEqual(distances[i, i].item(), 0.0, places=5)
        
        # Check that distances are symmetric
        for i in range(len(self.vocab)):
            for j in range(i+1, len(self.vocab)):
                self.assertAlmostEqual(distances[i, j].item(), distances[j, i].item(), places=5)
    
    def test_find_merge_candidates(self):
        """Test finding merge candidates."""
        # Set a high threshold to ensure some candidates are found
        self.tokenizer.merge_threshold = 10.0
        
        # Find merge candidates
        candidates = self.tokenizer._find_merge_candidates()
        
        # Check that candidates are tuples of (i, j, distance)
        for candidate in candidates:
            self.assertEqual(len(candidate), 3)
            i, j, dist = candidate
            self.assertIsInstance(i, int)
            self.assertIsInstance(j, int)
            self.assertIsInstance(dist, float)
            
            # Check that i and j are valid indices
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, len(self.vocab))
            self.assertGreaterEqual(j, 0)
            self.assertLess(j, len(self.vocab))
            
            # Check that the distance is below the threshold
            self.assertLessEqual(dist, self.tokenizer.merge_threshold)
    
    def test_merge_tokens(self):
        """Test merging tokens."""
        # Get initial vocabulary size
        initial_size = len(self.tokenizer.vocab)
        
        # Merge two tokens
        i, j = 4, 5  # 'a' and 'b'
        self.tokenizer._merge_tokens(i, j)
        
        # Check that the vocabulary size increased by 1
        self.assertEqual(len(self.tokenizer.vocab), initial_size + 1)
        
        # Check that the merged token is in the vocabulary
        merged_token = self.vocab[i] + self.vocab[j]
        self.assertIn(merged_token, self.tokenizer.vocab)
        
        # Check that the token to index mapping was updated
        self.assertEqual(self.tokenizer.token2idx[merged_token], initial_size)
        
        # Check that the embeddings tensor was extended
        self.assertEqual(self.tokenizer.embeddings.shape[0], initial_size + 1)
        
        # Check that the merge was recorded in the history
        self.assertEqual(len(self.tokenizer.merge_history), 1)
        self.assertEqual(self.tokenizer.merge_history[0], (self.vocab[i], self.vocab[j], merged_token))
    
    def test_tokenize_encode_decode(self):
        """Test tokenization, encoding, and decoding."""
        # Add some merged tokens to the vocabulary
        self.tokenizer.vocab.append("ab")
        self.tokenizer.token2idx["ab"] = len(self.tokenizer.vocab) - 1
        self.tokenizer.merge_history.append(("a", "b", "ab"))
        
        self.tokenizer.vocab.append("cd")
        self.tokenizer.token2idx["cd"] = len(self.tokenizer.vocab) - 1
        self.tokenizer.merge_history.append(("c", "d", "cd"))
        
        # Extend embeddings tensor
        new_embeddings = torch.zeros((len(self.tokenizer.vocab), self.embedding_dim + 1), dtype=torch.float32)
        new_embeddings[:self.embeddings.shape[0]] = self.embeddings
        new_embeddings[self.embeddings.shape[0]:] = self.embeddings[4:6]  # Use embeddings from 'a' and 'b'
        self.tokenizer.embeddings = torch.nn.Parameter(new_embeddings)
        
        # Test tokenization
        tokens = self.tokenizer.tokenize("abcde")
        self.assertEqual(tokens, ["ab", "cd", "e"])
        
        # Test encoding
        indices = self.tokenizer.encode("abcde")
        self.assertEqual(indices, [self.tokenizer.token2idx["ab"], self.tokenizer.token2idx["cd"], self.tokenizer.token2idx["e"]])
        
        # Test decoding
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, "abcde")
    
    def test_save_load(self):
        """Test saving and loading the tokenizer."""
        # Add some merged tokens to the vocabulary
        self.tokenizer.vocab.append("ab")
        self.tokenizer.token2idx["ab"] = len(self.tokenizer.vocab) - 1
        self.tokenizer.merge_history.append(("a", "b", "ab"))
        
        # Extend embeddings tensor
        new_embeddings = torch.zeros((len(self.tokenizer.vocab), self.embedding_dim + 1), dtype=torch.float32)
        new_embeddings[:self.embeddings.shape[0]] = self.embeddings
        new_embeddings[self.embeddings.shape[0]:] = self.embeddings[4:5]  # Use embeddings from 'a'
        self.tokenizer.embeddings = torch.nn.Parameter(new_embeddings)
        
        # Save the tokenizer
        save_path = os.path.join(self.temp_dir, "tokenizer")
        self.tokenizer.save(save_path)
        
        # Check that the files were created
        self.assertTrue(os.path.exists(os.path.join(save_path, "vocab.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "embeddings.pt")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "merges.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.json")))
        
        # Load the tokenizer
        loaded_tokenizer = HyperbolicTokenizer.load(save_path)
        
        # Check that the vocabulary was loaded correctly
        self.assertEqual(loaded_tokenizer.vocab, self.tokenizer.vocab)
        
        # Check that the embeddings were loaded correctly
        self.assertTrue(torch.allclose(loaded_tokenizer.embeddings, self.tokenizer.embeddings))
        
        # Check that the token to index mapping was loaded correctly
        for token, idx in self.tokenizer.token2idx.items():
            self.assertEqual(loaded_tokenizer.token2idx[token], idx)
        
        # Check that the parameters were loaded correctly
        self.assertEqual(loaded_tokenizer.curvature, self.tokenizer.curvature)
        self.assertEqual(loaded_tokenizer.merge_threshold, self.tokenizer.merge_threshold)
        
        # Check that the merge history was loaded correctly
        self.assertEqual(loaded_tokenizer.merge_history, self.tokenizer.merge_history)


if __name__ == "__main__":
    unittest.main()
