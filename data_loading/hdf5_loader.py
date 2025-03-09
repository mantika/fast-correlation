import h5py
import numpy as np
import torch
import os
import glob
from pathlib import Path


class HDF5DataLoader:
    """
    A data loader for efficient loading of large-scale HDF5 datasets.
    
    This loader handles loading data in batches to avoid memory issues with
    very large datasets, and supports moving data directly to GPU.
    """
    
    def __init__(self, file_path, dataset_key='data', dtype=torch.float32):
        """
        Initialize the HDF5 data loader.
        
        Args:
            file_path (str): Path to the HDF5 file.
            dataset_key (str): Key for the dataset in the HDF5 file.
            dtype (torch.dtype): Data type for the PyTorch tensors.
        """
        self.file_path = file_path
        self.dataset_key = dataset_key
        self.dtype = dtype
        
        # Open the file to get metadata but don't load data yet
        with h5py.File(file_path, 'r') as f:
            if dataset_key not in f:
                raise KeyError(f"Dataset key '{dataset_key}' not found in {file_path}")
            
            self.shape = f[dataset_key].shape
            self.n_samples = self.shape[0]
            self.n_features = self.shape[1]
        
        print(f"Dataset loaded: {self.n_samples} samples, {self.n_features} features")
    
    def validate_data(self):
        """
        Validate that the dataset meets requirements.
        
        Returns:
            bool: True if validation passes, raises exception otherwise.
        """
        if len(self.shape) != 2:
            raise ValueError(f"Expected 2D dataset, got {len(self.shape)}D")
        
        # Check for NaN values (sample a subset for large datasets)
        with h5py.File(self.file_path, 'r') as f:
            # Sample up to 1000 rows for validation
            sample_size = min(1000, self.n_samples)
            
            # Use sequential indices instead of random for more reliable testing
            sample_indices = np.arange(sample_size)
            sample_data = f[self.dataset_key][sample_indices, :]
            
            if np.isnan(sample_data).any():
                print("Warning: NaN values detected in the dataset. They will be replaced with zeros.")
        
        return True
    
    def load_all_data(self, device=None):
        """
        Load the entire dataset into memory.
        Only recommended for smaller datasets.
        
        Args:
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: The dataset as a PyTorch tensor.
        """
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.dataset_key][:]
            # Replace NaN values with zeros
            data = np.nan_to_num(data, nan=0.0)
            tensor_data = torch.tensor(data, dtype=self.dtype)
            
            if device is not None:
                tensor_data = tensor_data.to(device)
                
            return tensor_data
    
    def load_features(self, feature_indices, device=None):
        """
        Load specific features for all samples.
        
        Args:
            feature_indices (list): List of feature indices to load.
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: The selected features as a PyTorch tensor.
        """
        with h5py.File(self.file_path, 'r') as f:
            # Load data for all samples but only selected features
            data = f[self.dataset_key][:, feature_indices]
            # Replace NaN values with zeros
            data = np.nan_to_num(data, nan=0.0)
            tensor_data = torch.tensor(data, dtype=self.dtype)
            
            if device is not None:
                # Use non-blocking transfer for CUDA devices
                tensor_data = tensor_data.to(device, non_blocking=True)
                
            return tensor_data
    
    def load_batch(self, start_idx, end_idx, device=None):
        """
        Load a batch of features for all samples.
        
        Args:
            start_idx (int): Starting feature index.
            end_idx (int): Ending feature index (exclusive).
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: Batch of features as a PyTorch tensor.
        """
        return self.load_features(list(range(start_idx, end_idx)), device)
    
    def get_feature_batches(self, batch_size, n_gpus=1):
        """
        Generate feature batch indices.
        
        Args:
            batch_size (int): Number of features in each batch.
            n_gpus (int): Number of GPUs to use. The batch size will be adjusted
                to ensure even distribution across GPUs.
            
        Returns:
            list: List of (start_idx, end_idx) tuples for each batch.
        """
        # Adjust batch size to be divisible by number of GPUs
        if n_gpus > 1:
            batch_size = ((batch_size + n_gpus - 1) // n_gpus) * n_gpus
        
        batches = []
        for i in range(0, self.n_features, batch_size):
            end_idx = min(i + batch_size, self.n_features)
            batches.append((i, end_idx))
        return batches


class MultiFileHDF5DataLoader:
    """
    A data loader for handling multiple HDF5 files in a directory as a single dataset.
    
    This loader efficiently manages memory by only loading files as needed and
    supports distributed processing across multiple GPUs.
    """
    
    def __init__(self, directory_path, file_pattern="*.h5", dataset_key='data', dtype=torch.float32, max_files_in_memory=2):
        """
        Initialize the multi-file HDF5 data loader.
        
        Args:
            directory_path (str): Path to the directory containing HDF5 files.
            file_pattern (str): Glob pattern to match HDF5 files.
            dataset_key (str): Key for the dataset in the HDF5 files.
            dtype (torch.dtype): Data type for the PyTorch tensors.
            max_files_in_memory (int): Maximum number of files to load into memory at once.
        """
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.dataset_key = dataset_key
        self.dtype = dtype
        self.max_files_in_memory = max_files_in_memory
        
        # Find all matching files
        self.file_paths = sorted(glob.glob(os.path.join(directory_path, file_pattern)))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in {directory_path}")
        
        # Get metadata from all files
        self.file_metadata = []
        total_samples = 0
        n_features = None
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                if dataset_key not in f:
                    raise KeyError(f"Dataset key '{dataset_key}' not found in {file_path}")
                
                shape = f[dataset_key].shape
                if len(shape) != 2:
                    raise ValueError(f"Expected 2D dataset, got {len(shape)}D in {file_path}")
                
                # Ensure all files have the same number of features
                if n_features is None:
                    n_features = shape[1]
                elif n_features != shape[1]:
                    raise ValueError(f"Inconsistent number of features: {n_features} vs {shape[1]} in {file_path}")
                
                self.file_metadata.append({
                    'path': file_path,
                    'n_samples': shape[0],
                    'start_idx': total_samples,
                    'end_idx': total_samples + shape[0]
                })
                
                total_samples += shape[0]
        
        self.n_samples = total_samples
        self.n_features = n_features
        self.shape = (self.n_samples, self.n_features)
        
        print(f"Multi-file dataset loaded: {len(self.file_paths)} files, {self.n_samples} total samples, {self.n_features} features")
    
    def validate_data(self):
        """
        Validate that all datasets meet requirements.
        
        Returns:
            bool: True if validation passes, raises exception otherwise.
        """
        # Check a sample from each file for NaN values
        nan_detected = False
        
        for file_info in self.file_metadata:
            with h5py.File(file_info['path'], 'r') as f:
                # Sample up to 100 rows from each file
                sample_size = min(100, file_info['n_samples'])
                sample_indices = np.arange(sample_size)
                sample_data = f[self.dataset_key][sample_indices, :]
                
                if np.isnan(sample_data).any():
                    nan_detected = True
        
        if nan_detected:
            print("Warning: NaN values detected in the dataset. They will be replaced with zeros.")
        
        return True
    
    def _find_file_for_sample(self, sample_idx):
        """
        Find the file containing a specific sample.
        
        Args:
            sample_idx (int): Sample index in the combined dataset.
            
        Returns:
            dict: File metadata containing the sample.
        """
        for file_info in self.file_metadata:
            if file_info['start_idx'] <= sample_idx < file_info['end_idx']:
                return file_info
        
        raise IndexError(f"Sample index {sample_idx} out of range")
    
    def load_sample_range(self, start_sample, end_sample, feature_indices=None, device=None):
        """
        Load a range of samples for specific features.
        
        Args:
            start_sample (int): Starting sample index.
            end_sample (int): Ending sample index (exclusive).
            feature_indices (list, optional): List of feature indices to load.
                If None, all features are loaded.
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: The selected samples and features as a PyTorch tensor.
        """
        if start_sample < 0 or end_sample > self.n_samples:
            raise IndexError(f"Sample range {start_sample}-{end_sample} out of bounds (0-{self.n_samples})")
        
        if feature_indices is None:
            feature_indices = list(range(self.n_features))
        
        # Find which files contain the requested samples
        result_tensors = []
        current_idx = start_sample
        
        while current_idx < end_sample:
            file_info = self._find_file_for_sample(current_idx)
            
            # Calculate the range within this file
            file_start = max(0, current_idx - file_info['start_idx'])
            file_end = min(file_info['n_samples'], end_sample - file_info['start_idx'])
            
            # Load data from this file
            with h5py.File(file_info['path'], 'r') as f:
                if feature_indices is None:
                    data = f[self.dataset_key][file_start:file_end, :]
                else:
                    data = f[self.dataset_key][file_start:file_end, :][:, feature_indices]
                
                # Replace NaN values with zeros
                data = np.nan_to_num(data, nan=0.0)
                tensor_data = torch.tensor(data, dtype=self.dtype)
                result_tensors.append(tensor_data)
            
            # Update current index
            current_idx = file_info['end_idx']
        
        # Concatenate results
        if len(result_tensors) == 1:
            result = result_tensors[0]
        else:
            result = torch.cat(result_tensors, dim=0)
        
        if device is not None:
            result = result.to(device, non_blocking=True)
        
        return result
    
    def load_all_data(self, device=None):
        """
        Load the entire dataset into memory.
        Only recommended for smaller datasets.
        
        Args:
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: The dataset as a PyTorch tensor.
        """
        if len(self.file_paths) > self.max_files_in_memory:
            print(f"Warning: Loading {len(self.file_paths)} files into memory. This may cause memory issues.")
        
        return self.load_sample_range(0, self.n_samples, None, device)
    
    def load_features(self, feature_indices, device=None):
        """
        Load specific features for all samples.
        This loads data file by file to manage memory.
        
        Args:
            feature_indices (list): List of feature indices to load.
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: The selected features as a PyTorch tensor.
        """
        # Process files in batches to manage memory
        result_tensors = []
        files_per_batch = min(self.max_files_in_memory, len(self.file_paths))
        
        for i in range(0, len(self.file_paths), files_per_batch):
            batch_files = self.file_paths[i:i+files_per_batch]
            batch_start = self.file_metadata[i]['start_idx']
            batch_end = self.file_metadata[min(i+files_per_batch, len(self.file_paths))-1]['end_idx']
            
            batch_tensor = self.load_sample_range(batch_start, batch_end, feature_indices, device)
            result_tensors.append(batch_tensor)
            
            # Clear memory if device is None (CPU)
            if device is None:
                torch.cuda.empty_cache()
        
        # Concatenate results
        if len(result_tensors) == 1:
            return result_tensors[0]
        else:
            return torch.cat(result_tensors, dim=0)
    
    def load_batch(self, start_idx, end_idx, device=None):
        """
        Load a batch of features for all samples.
        
        Args:
            start_idx (int): Starting feature index.
            end_idx (int): Ending feature index (exclusive).
            device (torch.device): Device to move the data to (None for CPU).
            
        Returns:
            torch.Tensor: Batch of features as a PyTorch tensor.
        """
        return self.load_features(list(range(start_idx, end_idx)), device)
    
    def get_feature_batches(self, batch_size, n_gpus=1):
        """
        Generate feature batch indices.
        
        Args:
            batch_size (int): Number of features in each batch.
            n_gpus (int): Number of GPUs to use. The batch size will be adjusted
                to ensure even distribution across GPUs.
            
        Returns:
            list: List of (start_idx, end_idx) tuples for each batch.
        """
        # Adjust batch size to be divisible by number of GPUs
        if n_gpus > 1:
            batch_size = ((batch_size + n_gpus - 1) // n_gpus) * n_gpus
        
        batches = []
        for i in range(0, self.n_features, batch_size):
            end_idx = min(i + batch_size, self.n_features)
            batches.append((i, end_idx))
        return batches
    
    def get_sample_batches(self, batch_size):
        """
        Generate sample batch indices optimized for file access.
        This tries to create batches that align with file boundaries when possible.
        
        Args:
            batch_size (int): Number of samples in each batch.
            
        Returns:
            list: List of (start_idx, end_idx) tuples for each batch.
        """
        batches = []
        
        # First try to create batches that align with file boundaries
        current_idx = 0
        for file_info in self.file_metadata:
            file_start = file_info['start_idx']
            file_end = file_info['end_idx']
            file_samples = file_info['n_samples']
            
            # If file is smaller than batch_size, use the whole file
            if file_samples <= batch_size:
                batches.append((file_start, file_end))
                current_idx = file_end
            else:
                # Split the file into multiple batches
                for i in range(0, file_samples, batch_size):
                    start_idx = file_start + i
                    end_idx = min(start_idx + batch_size, file_end)
                    batches.append((start_idx, end_idx))
                current_idx = file_end
        
        return batches 