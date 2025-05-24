#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Wikipedia preprocessing functionality.
"""

import os
import sys
import unittest
import tempfile

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_wiki import clean_text, build_initial_vocab


class TestPreprocessWiki(unittest.TestCase):
    """Test cases for Wikipedia preprocessing functions."""
    
    def test_clean_text(self):
        """Test the text cleaning function."""
        # Test normalization to lowercase
        self.assertEqual(clean_text("Hello World"), "hello world")
        
        # Test special character removal
        self.assertEqual(clean_text("Hello, World!"), "hello, world")
        
        # Test unicode normalization
        self.assertEqual(clean_text("café"), "cafe")
        
        # Test whitespace normalization
        self.assertEqual(clean_text("hello   world"), "hello world")
        
        # Test combination of all
        self.assertEqual(clean_text("Hello, World! Special-Chars."), "hello, world special chars.")
    
    def test_build_initial_vocab(self):
        """Test building initial vocabulary."""
        # Create a temporary file with test content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("hello world\n")
            temp_file.write("this is a test\n")
            temp_file.write("hello again\n")
            temp_path = temp_file.name
        
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Build vocab with min_count=1
            build_initial_vocab(temp_path, output_path, min_count=1)
            
            # Read the resulting vocab
            with open(output_path, 'r') as f:
                vocab_lines = f.readlines()
            
            # Convert to list of tokens
            vocab = [line.strip() for line in vocab_lines]
            
            # Check if special tokens are included
            self.assertIn("<pad>", vocab)
            self.assertIn("<bos>", vocab)
            self.assertIn("<eos>", vocab)
            self.assertIn("<unk>", vocab)
            
            # Check if all characters from the input are included
            for char in "helo wrditasagn":
                self.assertIn(char, vocab)
            
            # Check that the space character is included
            self.assertIn(" ", vocab)
            
        finally:
            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
