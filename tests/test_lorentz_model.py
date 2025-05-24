#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Lorentz model operations in hyperbolic space.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path to import from embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.lorentz_model import (
    minkowski_dot, minkowski_norm, project_to_hyperboloid,
    exp_map, log_map, distance, parallel_transport
)


class TestLorentzModel(unittest.TestCase):
    """Test cases for Lorentz model operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a point at the origin of the Lorentz model
        self.origin = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        # Create some random points on the hyperboloid
        self.d = 3  # Dimensionality of the Euclidean part
        
        # Create random tangent vectors
        self.tangent_vecs = torch.randn(5, self.d)
        self.tangent_vecs = self.tangent_vecs / torch.norm(self.tangent_vecs, dim=1, keepdim=True) * 0.5
        
        # Map to hyperboloid
        self.points = torch.zeros(5, self.d + 1)
        for i, v in enumerate(self.tangent_vecs):
            p = exp_map(self.origin.unsqueeze(0), 
                       torch.cat([torch.zeros(1), v]).unsqueeze(0))[0]
            self.points[i] = p
        
        # Ensure all points are on the hyperboloid
        self.points = project_to_hyperboloid(self.points)
        
        # Curvature
        self.c = 1.0
    
    def test_minkowski_dot(self):
        """Test Minkowski dot product."""
        # The Minkowski dot product of the origin with itself should be 1
        dot_origin = minkowski_dot(self.origin, self.origin)
        self.assertAlmostEqual(dot_origin.item(), 1.0)
        
        # The Minkowski dot product of any point on the hyperboloid with itself should be 1
        for p in self.points:
            dot_p = minkowski_dot(p, p)
            self.assertAlmostEqual(dot_p.item(), 1.0, places=5)
        
        # The Minkowski dot product of two different points should be less than -1
        dot_diff = minkowski_dot(self.points[0], self.points[1])
        self.assertLess(dot_diff.item(), -1.0)
    
    def test_minkowski_norm(self):
        """Test Minkowski norm."""
        # The Minkowski norm of the origin should be 1
        norm_origin = minkowski_norm(self.origin)
        self.assertAlmostEqual(norm_origin.item(), 1.0)
        
        # The Minkowski norm of any point on the hyperboloid should be 1
        for p in self.points:
            norm_p = minkowski_norm(p)
            self.assertAlmostEqual(norm_p.item(), 1.0, places=5)
    
    def test_project_to_hyperboloid(self):
        """Test projection to hyperboloid."""
        # Generate random points
        random_points = torch.randn(10, self.d + 1)
        
        # Project to hyperboloid
        projected = project_to_hyperboloid(random_points)
        
        # Check that all projected points are on the hyperboloid
        for p in projected:
            # The Minkowski dot product of a point with itself should be 1
            dot_p = minkowski_dot(p, p)
            self.assertAlmostEqual(dot_p.item(), 1.0, places=5)
            
            # The first component should be positive
            self.assertGreater(p[0].item(), 0.0)
    
    def test_exp_map(self):
        """Test exponential map."""
        # The exp map of the zero tangent vector should be the origin
        zero_tangent = torch.zeros(self.d + 1)
        mapped = exp_map(self.origin.unsqueeze(0), zero_tangent.unsqueeze(0))[0]
        self.assertTrue(torch.allclose(mapped, self.origin, atol=1e-5))
        
        # The exp map should produce points on the hyperboloid
        for v in self.tangent_vecs:
            tangent = torch.cat([torch.zeros(1), v])
            mapped = exp_map(self.origin.unsqueeze(0), tangent.unsqueeze(0))[0]
            dot_mapped = minkowski_dot(mapped, mapped)
            self.assertAlmostEqual(dot_mapped.item(), 1.0, places=5)
    
    def test_log_map(self):
        """Test logarithmic map."""
        # The log map of the origin at the origin should be the zero vector
        log_origin = log_map(self.origin.unsqueeze(0), self.origin.unsqueeze(0))[0]
        self.assertTrue(torch.allclose(log_origin, torch.zeros_like(log_origin), atol=1e-5))
        
        # The log map should be the inverse of the exp map
        for v in self.tangent_vecs:
            tangent = torch.cat([torch.zeros(1), v])
            mapped = exp_map(self.origin.unsqueeze(0), tangent.unsqueeze(0))[0]
            log_mapped = log_map(self.origin.unsqueeze(0), mapped.unsqueeze(0))[0]
            self.assertTrue(torch.allclose(log_mapped, tangent, atol=1e-4))
    
    def test_distance(self):
        """Test distance computation."""
        # The distance from a point to itself should be 0
        for p in self.points:
            dist = distance(p.unsqueeze(0), p.unsqueeze(0))
            self.assertAlmostEqual(dist.item(), 0.0, places=5)
        
        # The distance between different points should be positive
        dist = distance(self.points[0].unsqueeze(0), self.points[1].unsqueeze(0))
        self.assertGreater(dist.item(), 0.0)
        
        # The distance should be symmetric
        dist_a_b = distance(self.points[0].unsqueeze(0), self.points[1].unsqueeze(0))
        dist_b_a = distance(self.points[1].unsqueeze(0), self.points[0].unsqueeze(0))
        self.assertAlmostEqual(dist_a_b.item(), dist_b_a.item(), places=5)
        
        # The distance should satisfy the triangle inequality
        for i in range(3):
            for j in range(i+1, 4):
                for k in range(j+1, 5):
                    dist_ij = distance(self.points[i].unsqueeze(0), self.points[j].unsqueeze(0))
                    dist_jk = distance(self.points[j].unsqueeze(0), self.points[k].unsqueeze(0))
                    dist_ik = distance(self.points[i].unsqueeze(0), self.points[k].unsqueeze(0))
                    
                    # Allow for small numerical errors
                    self.assertLessEqual(dist_ik.item(), dist_ij.item() + dist_jk.item() + 1e-4)
    
    def test_parallel_transport(self):
        """Test parallel transport."""
        # Create a tangent vector at a point
        p = self.points[0]
        q = self.points[1]
        
        # Create a tangent vector at p
        v = torch.cat([torch.zeros(1), self.tangent_vecs[2]])
        
        # Parallel transport v from p to q
        v_transported = parallel_transport(v.unsqueeze(0), p.unsqueeze(0), q.unsqueeze(0))[0]
        
        # The transported vector should be tangent to q
        dot_q_v = minkowski_dot(q, v_transported)
        self.assertAlmostEqual(dot_q_v.item(), 0.0, places=5)
        
        # The norm of the transported vector should be the same
        norm_v = torch.norm(v[1:])
        norm_v_transported = torch.norm(v_transported[1:])
        self.assertAlmostEqual(norm_v.item(), norm_v_transported.item(), places=5)


if __name__ == "__main__":
    unittest.main()
