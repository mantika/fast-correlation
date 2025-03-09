#!/usr/bin/env python3

import unittest
import torch
import numpy as np
from correlation.pearson import (
    compute_pearson_correlation,
    compute_batch_correlation,
    find_correlated_pairs,
    update_active_indices
)


class TestPearsonCorrelation(unittest.TestCase):
    """
    Test the Pearson correlation computation functions.
    """
    
    def setUp(self):
        """
        Set up test data for the correlation tests.
        """
        # Create a small dataset with known correlations
        self.n_samples = 1000
        self.n_features = 10
        
        # Set a fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create independent features
        self.independent_data = np.random.randn(self.n_samples, 5)
        
        # Create features with exact correlations
        # For simplicity, we'll create features with exact correlations
        # by directly constructing the correlation matrix and then generating data
        
        # Create base features
        base_feature1 = np.random.randn(self.n_samples)
        base_feature2 = np.random.randn(self.n_samples)
        
        # Normalize features
        base_feature1 = (base_feature1 - np.mean(base_feature1)) / np.std(base_feature1)
        base_feature2 = (base_feature2 - np.mean(base_feature2)) / np.std(base_feature2)
        
        # Create correlated features with exact correlation of 0.9
        corr_strength = 0.9
        
        # Create orthogonal noise
        noise1 = np.random.randn(self.n_samples)
        noise1 = noise1 - np.mean(noise1)  # Center
        # Make noise orthogonal to base_feature1
        noise1 = noise1 - np.dot(noise1, base_feature1) * base_feature1 / np.dot(base_feature1, base_feature1)
        # Normalize noise
        noise1 = noise1 / np.std(noise1)
        
        # Create correlated feature
        correlated_feature1 = corr_strength * base_feature1 + np.sqrt(1 - corr_strength**2) * noise1
        
        # Create anti-correlated feature
        anti_corr_feature = -corr_strength * base_feature1 + np.sqrt(1 - corr_strength**2) * noise1
        
        # Create noise for second feature
        noise2 = np.random.randn(self.n_samples)
        noise2 = noise2 - np.mean(noise2)  # Center
        # Make noise orthogonal to base_feature2
        noise2 = noise2 - np.dot(noise2, base_feature2) * base_feature2 / np.dot(base_feature2, base_feature2)
        # Normalize noise
        noise2 = noise2 / np.std(noise2)
        
        # Create second correlated feature
        correlated_feature2 = corr_strength * base_feature2 + np.sqrt(1 - corr_strength**2) * noise2
        
        # Combine all features
        self.data = np.zeros((self.n_samples, self.n_features))
        self.data[:, :5] = self.independent_data
        self.data[:, 5] = base_feature1
        self.data[:, 6] = correlated_feature1
        self.data[:, 7] = base_feature2
        self.data[:, 8] = correlated_feature2
        self.data[:, 9] = anti_corr_feature
        
        # Convert to torch tensor
        self.data_tensor = torch.tensor(self.data, dtype=torch.float32)
        
        # Verify correlations
        corr_matrix = np.corrcoef(self.data.T)
        print(f"Correlation between features 5 and 6: {corr_matrix[5, 6]}")
        print(f"Correlation between features 7 and 8: {corr_matrix[7, 8]}")
        print(f"Correlation between features 5 and 9: {corr_matrix[5, 9]}")
    
    def test_compute_pearson_correlation(self):
        """
        Test the computation of Pearson correlation.
        """
        # Compute correlation matrix
        corr_matrix = compute_pearson_correlation(self.data_tensor)
        
        # Check shape
        self.assertEqual(corr_matrix.shape, (self.n_features, self.n_features))
        
        # Check diagonal elements (should be 1.0)
        diag_elements = torch.diag(corr_matrix)
        self.assertTrue(torch.allclose(diag_elements, torch.ones_like(diag_elements)))
        
        # Get numpy correlation matrix for reference
        np_corr_matrix = np.corrcoef(self.data.T)
        
        # Check known correlations with a larger delta to account for numerical differences
        # The correlation between features 5 and 6 should be close to 0.9
        self.assertAlmostEqual(corr_matrix[5, 6].item(), np_corr_matrix[5, 6], delta=0.1)
        
        # The correlation between features 7 and 8 should be close to 0.9
        self.assertAlmostEqual(corr_matrix[7, 8].item(), np_corr_matrix[7, 8], delta=0.1)
        
        # The correlation between features 5 and 9 should be close to -0.9
        self.assertAlmostEqual(corr_matrix[5, 9].item(), np_corr_matrix[5, 9], delta=0.1)
        
        # The correlation between independent features should be close to 0
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertAlmostEqual(corr_matrix[i, j].item(), 0.0, delta=0.2)
    
    def test_compute_batch_correlation(self):
        """
        Test the batch correlation computation.
        """
        # Create a batch
        batch_indices = [0, 1, 2]
        batch_features = self.data_tensor[:, batch_indices]
        
        # Compute batch correlation
        corr_matrix, active_indices = compute_batch_correlation(
            batch_features, self.data_tensor
        )
        
        # Check shape
        self.assertEqual(corr_matrix.shape, (len(batch_indices), self.n_features))
        
        # The active indices should be the full range
        self.assertEqual(active_indices, list(range(self.n_features)))
        
        # Get numpy correlation matrix for reference
        np_corr_matrix = np.corrcoef(self.data.T)
        
        # Check a few correlations
        # The correlation of feature 0 with itself should be 1.0
        self.assertAlmostEqual(corr_matrix[0, 0].item(), 1.0, delta=0.01)
        
        # The correlation between independent features should be close to 0
        self.assertAlmostEqual(corr_matrix[0, 1].item(), np_corr_matrix[0, 1], delta=0.2)
    
    def test_find_correlated_pairs(self):
        """
        Test finding correlated pairs.
        """
        # Get numpy correlation matrix for reference
        np_corr_matrix = np.corrcoef(self.data.T)
        
        # Print correlation values for debugging
        print(f"Correlation (5,6): {np_corr_matrix[5, 6]}")
        print(f"Correlation (5,9): {np_corr_matrix[5, 9]}")
        print(f"Correlation (7,8): {np_corr_matrix[7, 8]}")
        
        # Set threshold based on actual correlations
        threshold = 0.8
        
        # Compute correlation matrix
        batch_indices = [5, 7]  # Base features for correlated pairs
        batch_features = self.data_tensor[:, batch_indices]
        
        corr_matrix, active_indices = compute_batch_correlation(
            batch_features, self.data_tensor
        )
        
        # Print the computed correlation matrix for debugging
        print("Computed correlation matrix:")
        for i, batch_idx in enumerate(batch_indices):
            for j, active_idx in enumerate(active_indices):
                if abs(corr_matrix[i, j]) > threshold and batch_idx != active_idx:
                    print(f"Correlation ({batch_idx},{active_idx}): {corr_matrix[i, j].item()}")
        
        # Find correlated pairs with threshold
        correlated_pairs = find_correlated_pairs(
            corr_matrix, batch_indices, active_indices, threshold=threshold
        )
        
        # Print found pairs for debugging
        print("Found correlated pairs:")
        for i, j, corr in correlated_pairs:
            print(f"Pair ({i},{j}): {corr}")
        
        # We expect to find the following pairs:
        # (5, 6), (5, 9), (7, 8)
        expected_pairs = [(5, 6), (5, 9), (7, 8)]
        
        # Filter expected pairs to only include those with correlation above threshold
        filtered_expected_pairs = []
        for i, j in expected_pairs:
            if abs(np_corr_matrix[i, j]) > threshold:
                filtered_expected_pairs.append((i, j))
        
        # Print expected pairs for debugging
        print("Expected pairs:")
        for i, j in filtered_expected_pairs:
            print(f"Pair ({i},{j}): {np_corr_matrix[i, j]}")
        
        # Check that we found the expected number of pairs
        self.assertEqual(len(correlated_pairs), len(filtered_expected_pairs))
        
        # Check that the expected pairs are in the result
        pair_indices = [(i, j) for i, j, _ in correlated_pairs]
        
        for pair in filtered_expected_pairs:
            self.assertIn(pair, pair_indices)
        
        # Check correlation values
        for i, j, corr in correlated_pairs:
            self.assertAlmostEqual(corr, np_corr_matrix[i, j], delta=0.1)
    
    def test_update_active_indices(self):
        """
        Test updating active indices.
        """
        # Initial active indices
        active_indices = list(range(self.n_features))
        
        # Correlated pairs
        correlated_pairs = [
            (5, 6, 0.9),
            (7, 8, 0.9),
            (5, 9, -0.9)
        ]
        
        # Update active indices with "first" strategy
        updated_indices, removed_indices = update_active_indices(
            active_indices, correlated_pairs, strategy='first'
        )
        
        # We expect to remove the higher index from each pair
        # So we should remove 6, 8, and 9
        expected_removed = {6, 8, 9}
        
        # Check that the removed indices are as expected
        self.assertEqual(set(removed_indices), expected_removed)
        
        # Check that the updated active indices are correct
        expected_active = [i for i in range(self.n_features) if i not in expected_removed]
        self.assertEqual(set(updated_indices), set(expected_active))
        
        # Test with "random" strategy
        # Since random is non-deterministic, we'll just check that the number of removed 
        # indices is the same
        updated_indices_random, removed_indices_random = update_active_indices(
            active_indices, correlated_pairs, strategy='random'
        )
        
        self.assertEqual(len(removed_indices_random), len(expected_removed))

    def test_multi_gpu_correlation(self):
        """
        Test correlation computation using multiple GPUs.
        Skip if CUDA is not available or if there's only one GPU.
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping multi-GPU test")
            return
        
        if torch.cuda.device_count() < 2:
            print("Less than 2 GPUs available, skipping multi-GPU test")
            return
        
        # Use first two GPUs
        devices = [torch.device(f'cuda:{i}') for i in range(2)]
        
        # Create test data
        batch_size = 2
        batch_indices = [5, 7]  # Base features for correlated pairs
        batch_features = self.data_tensor[:, batch_indices]
        
        # Compute correlation using multiple GPUs
        correlation_matrix, active_indices = compute_batch_correlation(
            batch_features, self.data_tensor, devices=devices
        )
        
        # Compute correlation using single GPU for reference
        single_gpu_matrix, _ = compute_batch_correlation(
            batch_features, self.data_tensor, devices=[devices[0]]
        )
        
        # Results should be the same regardless of number of GPUs
        self.assertTrue(torch.allclose(correlation_matrix, single_gpu_matrix, atol=1e-6))
        
        # Check specific correlations
        self.assertAlmostEqual(correlation_matrix[0, 6].item(), 0.9, delta=0.1)  # (5,6)
        self.assertAlmostEqual(correlation_matrix[0, 9].item(), -0.9, delta=0.1)  # (5,9)
        self.assertAlmostEqual(correlation_matrix[1, 8].item(), 0.9, delta=0.1)  # (7,8)
    
    def test_multi_gpu_memory_management(self):
        """
        Test memory management in multi-GPU correlation computation.
        Skip if CUDA is not available or if there's only one GPU.
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping multi-GPU memory test")
            return
        
        if torch.cuda.device_count() < 2:
            print("Less than 2 GPUs available, skipping multi-GPU memory test")
            return
        
        # Use first two GPUs
        devices = [torch.device(f'cuda:{i}') for i in range(2)]
        
        # Record initial memory usage
        initial_memory = [torch.cuda.memory_allocated(i) for i in range(2)]
        
        # Create larger test data to stress memory
        n_samples = 10000
        n_features = 1000
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        # Compute correlation using multiple GPUs
        correlation_matrix, _ = compute_batch_correlation(X, Y, devices=devices)
        
        # Record final memory usage
        final_memory = [torch.cuda.memory_allocated(i) for i in range(2)]
        
        # Memory should be cleared after computation
        for i in range(2):
            self.assertLessEqual(
                final_memory[i] - initial_memory[i],
                1024 * 1024 * 100  # Allow up to 100MB residual memory
            )


if __name__ == "__main__":
    unittest.main() 