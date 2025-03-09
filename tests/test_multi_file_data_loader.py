#!/usr/bin/env python3

import unittest
import os
import h5py
import numpy as np
import torch
import tempfile
import shutil
from data_loading import MultiFileHDF5DataLoader


class TestMultiFileHDF5DataLoader(unittest.TestCase):
    """
    Test the MultiFileHDF5DataLoader for handling multiple HDF5 files.
    """
    
    def setUp(self):
        """
        Set up test data for the multi-file data loader tests.
        Create multiple HDF5 files in a temporary directory.
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Parameters for test data
        self.n_files = 3
        self.n_samples_per_file = [50, 70, 60]  # Different number of samples per file
        self.n_features = 20
        self.total_samples = sum(self.n_samples_per_file)
        
        # Generate random data and create files
        np.random.seed(42)
        self.file_paths = []
        self.all_data = []
        
        for i in range(self.n_files):
            file_path = os.path.join(self.temp_dir, f"test_data_{i}.h5")
            self.file_paths.append(file_path)
            
            # Generate data for this file
            data = np.random.randn(self.n_samples_per_file[i], self.n_features)
            
            # Add some NaN values for testing
            if i == 1:  # Add NaNs to the second file
                data[10, 5] = np.nan
                data[20, 15] = np.nan
            
            # Save the dataset to HDF5
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=data)
                # Add another dataset to test key selection
                f.create_dataset('other_data', data=np.zeros((10, 10)))
            
            self.all_data.append(data)
        
        # Concatenate all data for reference
        self.combined_data = np.vstack(self.all_data)
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """
        Test initialization of the multi-file data loader.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Check attributes
        self.assertEqual(loader.n_samples, self.total_samples)
        self.assertEqual(loader.n_features, self.n_features)
        self.assertEqual(loader.shape, (self.total_samples, self.n_features))
        self.assertEqual(len(loader.file_paths), self.n_files)
    
    def test_validate_data(self):
        """
        Test data validation across multiple files.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Validation should pass
        self.assertTrue(loader.validate_data())
    
    def test_load_all_data(self):
        """
        Test loading all data from multiple files.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Load data
        data = loader.load_all_data()
        
        # Check shape
        self.assertEqual(data.shape, (self.total_samples, self.n_features))
        
        # Check data type
        self.assertEqual(data.dtype, torch.float32)
        
        # Check that NaN values are replaced with zeros
        # The NaN values were in the second file, so we need to calculate their global indices
        nan_sample_1 = self.n_samples_per_file[0] + 10
        nan_sample_2 = self.n_samples_per_file[0] + 20
        self.assertEqual(data[nan_sample_1, 5].item(), 0.0)
        self.assertEqual(data[nan_sample_2, 15].item(), 0.0)
        
        # Check a few values against the combined reference data
        np_data = np.nan_to_num(self.combined_data, nan=0.0)
        for i in range(0, self.total_samples, 10):
            for j in range(0, self.n_features, 5):
                self.assertAlmostEqual(data[i, j].item(), np_data[i, j], delta=1e-5)
    
    def test_load_features(self):
        """
        Test loading specific features from multiple files.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Load specific features
        feature_indices = [0, 5, 10, 15]
        data = loader.load_features(feature_indices)
        
        # Check shape
        self.assertEqual(data.shape, (self.total_samples, len(feature_indices)))
        
        # Check a few values
        np_data = np.nan_to_num(self.combined_data, nan=0.0)
        for i in range(0, self.total_samples, 10):
            for j_idx, j in enumerate(feature_indices):
                self.assertAlmostEqual(data[i, j_idx].item(), np_data[i, j], delta=1e-5)
    
    def test_load_batch(self):
        """
        Test loading a batch of features from multiple files.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Load a batch
        start_idx = 5
        end_idx = 10
        data = loader.load_batch(start_idx, end_idx)
        
        # Check shape
        self.assertEqual(data.shape, (self.total_samples, end_idx - start_idx))
        
        # Check a few values
        np_data = np.nan_to_num(self.combined_data, nan=0.0)
        for i in range(0, self.total_samples, 10):
            for j_idx, j in enumerate(range(start_idx, end_idx)):
                self.assertAlmostEqual(data[i, j_idx].item(), np_data[i, j], delta=1e-5)
    
    def test_load_sample_range(self):
        """
        Test loading a range of samples across file boundaries.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Load samples that span across files
        start_sample = self.n_samples_per_file[0] - 10  # 10 samples before end of first file
        end_sample = self.n_samples_per_file[0] + 10    # 10 samples into second file
        data = loader.load_sample_range(start_sample, end_sample)
        
        # Check shape
        self.assertEqual(data.shape, (end_sample - start_sample, self.n_features))
        
        # Check a few values
        np_data = np.nan_to_num(self.combined_data, nan=0.0)
        for i in range(start_sample, end_sample, 2):
            for j in range(0, self.n_features, 5):
                self.assertAlmostEqual(data[i - start_sample, j].item(), np_data[i, j], delta=1e-5)
    
    def test_get_feature_batches(self):
        """
        Test generating feature batches.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
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
    
    def test_get_sample_batches(self):
        """
        Test generating sample batches optimized for file access.
        """
        loader = MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='data')
        
        # Generate batches with different batch sizes
        batch_size = 30  # Smaller than some files, larger than others
        batches = loader.get_sample_batches(batch_size)
        
        # Check that all samples are covered
        total_samples_in_batches = sum(end - start for start, end in batches)
        self.assertEqual(total_samples_in_batches, self.total_samples)
        
        # Check that batches don't overlap
        all_samples = []
        for start, end in batches:
            batch_samples = list(range(start, end))
            # Check no duplicates
            self.assertEqual(len(set(batch_samples) & set(all_samples)), 0)
            all_samples.extend(batch_samples)
        
        # Check that all samples are included
        self.assertEqual(len(all_samples), self.total_samples)
        self.assertEqual(set(all_samples), set(range(self.total_samples)))
    
    def test_invalid_key(self):
        """
        Test loading a dataset with an invalid key.
        """
        # This should raise a KeyError
        with self.assertRaises(KeyError):
            MultiFileHDF5DataLoader(self.temp_dir, file_pattern="test_data_*.h5", dataset_key='invalid_key')
    
    def test_memory_management(self):
        """
        Test that memory management works correctly with max_files_in_memory.
        """
        # Create a loader with max_files_in_memory=1
        loader = MultiFileHDF5DataLoader(
            self.temp_dir, 
            file_pattern="test_data_*.h5", 
            dataset_key='data',
            max_files_in_memory=1
        )
        
        # Load all features
        feature_indices = list(range(self.n_features))
        data = loader.load_features(feature_indices)
        
        # Check shape
        self.assertEqual(data.shape, (self.total_samples, self.n_features))
        
        # Check a few values
        np_data = np.nan_to_num(self.combined_data, nan=0.0)
        for i in range(0, self.total_samples, 10):
            for j in range(0, self.n_features, 5):
                self.assertAlmostEqual(data[i, j].item(), np_data[i, j], delta=1e-5)


if __name__ == "__main__":
    unittest.main() 