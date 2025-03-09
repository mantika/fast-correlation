import h5py
import numpy as np
import torch


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