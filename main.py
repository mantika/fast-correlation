#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

from config import Config, parse_args
from data_loading import HDF5DataLoader, MultiFileHDF5DataLoader
from correlation.pearson import (
    compute_batch_correlation,
    find_correlated_pairs,
    update_active_indices
)
from utils.gpu import get_available_gpus, setup_multi_gpu, print_gpu_info
from utils.checkpoint import save_checkpoint, load_checkpoint, save_results
from utils.progress import ProgressTracker


def initialize_gpu(config):
    """
    Initialize GPU devices based on configuration.
    
    Args:
        config (Config): Configuration object.
        
    Returns:
        list: List of PyTorch device objects.
    """
    if not config.use_gpu or not torch.cuda.is_available():
        print("Using CPU for computation")
        return [torch.device('cpu')]
    
    # Print GPU information
    print_gpu_info()
    
    # Set up GPU devices
    devices = setup_multi_gpu(config.gpu_ids)
    
    if not devices or devices[0].type == 'cpu':
        print("No valid GPUs found, falling back to CPU")
        return [torch.device('cpu')]
    
    device_str = ", ".join([f"GPU {d.index}" for d in devices if d.type == 'cuda'])
    print(f"Using {device_str} for computation")
    
    return devices


def load_data_and_validate(config, device):
    """
    Load the dataset and validate it.
    
    Args:
        config (Config): Configuration object.
        device (torch.device): Device to load data to.
        
    Returns:
        HDF5DataLoader or MultiFileHDF5DataLoader: Data loader object.
    """
    if config.input_dir:
        print(f"Loading multi-file dataset from directory {config.input_dir}")
        loader = MultiFileHDF5DataLoader(
            config.input_dir, 
            file_pattern=config.file_pattern,
            dataset_key=config.dataset_key,
            max_files_in_memory=config.max_files_in_memory
        )
    else:
        print(f"Loading dataset from {config.input_file}")
        loader = HDF5DataLoader(config.input_file, config.dataset_key)
    
    # Validate the dataset
    loader.validate_data()
    
    return loader


def find_correlated_features(config, data_loader, devices):
    """
    Find correlated features in the dataset using multiple GPUs if available.
    
    Args:
        config (Config): Configuration object.
        data_loader (HDF5DataLoader or MultiFileHDF5DataLoader): Data loader object.
        devices (list): List of PyTorch device objects.
        
    Returns:
        list: List of correlated feature pairs.
    """
    # Initialize active indices and correlated pairs
    active_indices = list(range(data_loader.n_features))
    correlated_pairs = []
    
    # Check if resuming from a checkpoint
    start_batch_idx = 0
    elapsed_time = 0
    
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Resuming from checkpoint {config.resume_from}")
        active_indices, prev_pairs, start_batch_idx, elapsed_time = load_checkpoint(config.resume_from)
        correlated_pairs.extend(prev_pairs)
        print(f"Loaded {len(active_indices)} active indices and {len(prev_pairs)} correlated pairs")
        print(f"Resuming from batch {start_batch_idx}")
    
    # Initialize progress tracker
    progress = ProgressTracker(
        total_features=data_loader.n_features,
        batch_size=config.batch_size,
        checkpoint_freq=config.checkpoint_freq
    )
    
    # Adjust start time if resuming
    if elapsed_time > 0:
        progress.start_time = time.time() - elapsed_time
    
    # Generate batches
    batches = data_loader.get_feature_batches(config.batch_size, n_gpus=len(devices))
    
    # Skip batches that have already been processed
    batches = batches[start_batch_idx:]
    
    # Process each batch
    for batch_idx, (start_idx, end_idx) in enumerate(batches, start=start_batch_idx):
        # Start batch timing
        progress.start_batch()
        
        # Load batch features
        batch_indices = list(range(start_idx, end_idx))
        batch_features = data_loader.load_batch(start_idx, end_idx)
        
        # Load all active features
        active_feature_indices = [idx for idx in active_indices if idx not in batch_indices]
        all_features = data_loader.load_features(active_feature_indices)
        
        # Compute correlation matrix for the current batch using all available GPUs
        correlation_matrix, _ = compute_batch_correlation(
            batch_features, all_features, active_indices=active_feature_indices, devices=devices
        )
        
        # Find correlated pairs
        batch_correlated_pairs = find_correlated_pairs(
            correlation_matrix, batch_indices, active_feature_indices, config.threshold
        )
        
        # Add new correlated pairs to the list
        correlated_pairs.extend(batch_correlated_pairs)
        
        # Update active indices
        active_indices, removed_indices = update_active_indices(
            active_indices, batch_correlated_pairs, strategy=config.pruning_strategy
        )
        
        # Get GPU memory usage (max across all devices)
        gpu_memory = max(progress.get_gpu_memory_usage(device) for device in devices)
        
        # End batch timing and update progress
        should_checkpoint = progress.end_batch(
            n_active=len(active_indices),
            n_pruned=len(removed_indices),
            n_pairs=len(batch_correlated_pairs),
            gpu_memory=gpu_memory
        )
        
        # Save checkpoint if needed
        if should_checkpoint:
            checkpoint_file = save_checkpoint(
                config.checkpoint_dir,
                active_indices,
                correlated_pairs,
                batch_idx + 1,  # Next batch index
                progress.get_elapsed_time()
            )
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()
    
    # Close progress tracker
    progress.close()
    
    return correlated_pairs


def main():
    """
    Main entry point for the correlation computation.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create configuration
    config = Config.from_args(args)
    
    # Ensure checkpoint directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize GPU
    devices = initialize_gpu(config)
    
    # Load dataset
    data_loader = load_data_and_validate(config, devices[0])
    
    # Find correlated features
    correlated_pairs = find_correlated_features(config, data_loader, devices)
    
    # Save results
    metadata = {
        "n_samples": data_loader.n_samples,
        "n_features": data_loader.n_features,
        "threshold": config.threshold,
        "n_correlated_pairs": len(correlated_pairs)
    }
    
    save_results(config.output_file, correlated_pairs, metadata)
    
    print(f"Found {len(correlated_pairs)} correlated feature pairs")
    print(f"Results saved to {config.output_file}")


if __name__ == "__main__":
    main() 