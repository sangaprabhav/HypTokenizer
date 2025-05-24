#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for WordNet graph construction functionality.
"""

import os
import sys
import unittest
import networkx as nx
import tempfile

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_wordnet_graph import build_wordnet_graph


class TestWordNetGraph(unittest.TestCase):
    """Test cases for WordNet graph construction."""
    
    def test_build_wordnet_graph(self):
        """Test building the WordNet graph."""
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix='.gpk', delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Build the graph
            G = build_wordnet_graph(output_path)
            
            # Check that the graph is not empty
            self.assertGreater(G.number_of_nodes(), 0)
            self.assertGreater(G.number_of_edges(), 0)
            
            # Check that the graph file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Load the graph from file
            G_loaded = nx.read_gpickle(output_path)
            
            # Check that the loaded graph is the same
            self.assertEqual(G.number_of_nodes(), G_loaded.number_of_nodes())
            self.assertEqual(G.number_of_edges(), G_loaded.number_of_edges())
            
            # Check graph properties
            for node in list(G.nodes())[:10]:  # Check first 10 nodes
                # Check that node is a string
                self.assertIsInstance(node, str)
                
                # Check the format of node names (synset name format)
                self.assertIn('.', node)
                
            # Check that the graph has hypernym relationships
            # Find a node with neighbors
            for node in G.nodes():
                if len(list(G.neighbors(node))) > 0:
                    # Found a node with neighbors
                    break
            
            # Check that the node has neighbors
            self.assertGreater(len(list(G.neighbors(node))), 0)
            
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
