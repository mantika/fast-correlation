#!/usr/bin/env python3

import unittest
import os
import h5py
import numpy as np
import torch
import tempfile
from data_loading import HDF5DataLoader


class TestHDF5DataLoader(unittest.TestCase):
    """
    Test the HDF5 data loader.
    """
    
    def setUp(self):
        """
        Set up test data for the data loader tests.
        """
        # Create a temporary file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test_data.h5")
        
        # Create a small dataset
        self.n_samples = 100
        self.n_features = 20
        
        # Generate random data
        np.random.seed(42)
        self.data = np.random.randn(self.n_samples, self.n_features)
        
        # Add some NaN values for testing
        self.data[10, 5] = np.nan
        self.data[20, 15] = np.nan
        
        # Save the dataset to HDF5
        with h5py.File(self.temp_file, 'w') as f:
            f.create_dataset('data', data=self.data)
            f.create_dataset('other_data', data=np.zeros((10, 10)))
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        self.temp_dir.cleanup()
    
    def test_init(self):
        """
        Test initialization of the data loader.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Check attributes
        self.assertEqual(loader.n_samples, self.n_samples)
        self.assertEqual(loader.n_features, self.n_features)
        self.assertEqual(loader.shape, (self.n_samples, self.n_features))
    
    def test_validate_data(self):
        """
        Test data validation.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Validation should pass
        self.assertTrue(loader.validate_data())
    
    def test_load_all_data(self):
        """
        Test loading all data at once.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Load data
        data = loader.load_all_data()
        
        # Check shape
        self.assertEqual(data.shape, (self.n_samples, self.n_features))
        
        # Check data type
        self.assertEqual(data.dtype, torch.float32)
        
        # Check that NaN values are replaced with zeros
        self.assertEqual(data[10, 5].item(), 0.0)
        self.assertEqual(data[20, 15].item(), 0.0)
        
        # Check a few values
        np_data = self.data.copy()
        np_data = np.nan_to_num(np_data, nan=0.0)
        for i in range(0, self.n_samples, 10):
            for j in range(0, self.n_features, 5):
                self.assertAlmostEqual(data[i, j].item(), np_data[i, j], delta=1e-5)
    
    def test_load_features(self):
        """
        Test loading specific features.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Load specific features
        feature_indices = [0, 5, 10, 15]
        data = loader.load_features(feature_indices)
        
        # Check shape
        self.assertEqual(data.shape, (self.n_samples, len(feature_indices)))
        
        # Check a few values
        np_data = self.data.copy()
        np_data = np.nan_to_num(np_data, nan=0.0)
        for i in range(0, self.n_samples, 10):
            for j_idx, j in enumerate(feature_indices):
                self.assertAlmostEqual(data[i, j_idx].item(), np_data[i, j], delta=1e-5)
    
    def test_load_batch(self):
        """
        Test loading a batch of features.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Load a batch
        start_idx = 5
        end_idx = 10
        data = loader.load_batch(start_idx, end_idx)
        
        # Check shape
        self.assertEqual(data.shape, (self.n_samples, end_idx - start_idx))
        
        # Check a few values
        np_data = self.data.copy()
        np_data = np.nan_to_num(np_data, nan=0.0)
        for i in range(0, self.n_samples, 10):
            for j_idx, j in enumerate(range(start_idx, end_idx)):
                self.assertAlmostEqual(data[i, j_idx].item(), np_data[i, j], delta=1e-5)
    
    def test_get_feature_batches(self):
        """
        Test generating feature batches.
        """
        loader = HDF5DataLoader(self.temp_file, 'data')
        
        # Generate batches with different batch sizes
        batch_size = 5
        batches = loader.get_feature_batches(batch_size)
        
        # Check number of batches
        expected_n_batches = (self.n_features + batch_size - 1) // batch_size
        self.assertEqual(len(batches), expected_n_batches)
        
        # Check batch indices
        for i, (start_idx, end_idx) in enumerate(batches):
            self.assertEqual(start_idx, i * batch_size)
            self.assertEqual(end_idx, min((i + 1) * batch_size, self.n_features))
    
    def test_invalid_key(self):
        """
        Test loading a dataset with an invalid key.
        """
        # This should raise a KeyError
        with self.assertRaises(KeyError):
            HDF5DataLoader(self.temp_file, 'invalid_key')


if __name__ == "__main__":
    unittest.main() 