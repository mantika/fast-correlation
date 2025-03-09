import json
import os
import torch
import numpy as np
import h5py
from datetime import datetime


def save_checkpoint(output_dir, active_indices, correlated_pairs, batch_idx, elapsed_time, prefix="checkpoint"):
    """
    Save a checkpoint of the correlation computation progress.
    
    Args:
        output_dir (str): Directory to save the checkpoint.
        active_indices (list): List of active feature indices.
        correlated_pairs (list): List of correlated feature pairs.
        batch_idx (int): Index of the current batch.
        elapsed_time (float): Elapsed time in seconds.
        prefix (str): Prefix for the checkpoint file.
        
    Returns:
        str: Path to the saved checkpoint file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint dictionary
    checkpoint = {
        "timestamp": timestamp,
        "batch_idx": batch_idx,
        "elapsed_time": elapsed_time,
        "active_indices": active_indices,
        "n_correlated_pairs": len(correlated_pairs)
    }
    
    # Save checkpoint metadata
    checkpoint_file = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Save correlated pairs to a separate file (could be large)
    pairs_file = os.path.join(output_dir, f"{prefix}_{timestamp}_pairs.h5")
    with h5py.File(pairs_file, 'w') as f:
        # Convert correlated pairs to a structured array
        if correlated_pairs:
            dtype = np.dtype([
                ('feature_i', np.int32),
                ('feature_j', np.int32),
                ('correlation', np.float32)
            ])
            pairs_array = np.array(
                [(i, j, corr) for i, j, corr in correlated_pairs],
                dtype=dtype
            )
            f.create_dataset('correlated_pairs', data=pairs_array)
    
    print(f"Checkpoint saved to {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_file):
    """
    Load a checkpoint of the correlation computation progress.
    
    Args:
        checkpoint_file (str): Path to the checkpoint file.
        
    Returns:
        tuple: (active_indices, correlated_pairs, batch_idx, elapsed_time)
    """
    # Load checkpoint metadata
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    active_indices = checkpoint["active_indices"]
    batch_idx = checkpoint["batch_idx"]
    elapsed_time = checkpoint["elapsed_time"]
    
    # Load correlated pairs from the separate file
    pairs_file = checkpoint_file.replace(".json", "_pairs.h5")
    correlated_pairs = []
    
    if os.path.exists(pairs_file):
        with h5py.File(pairs_file, 'r') as f:
            if 'correlated_pairs' in f:
                pairs_array = f['correlated_pairs'][:]
                correlated_pairs = [
                    (int(row['feature_i']), int(row['feature_j']), float(row['correlation']))
                    for row in pairs_array
                ]
    
    return active_indices, correlated_pairs, batch_idx, elapsed_time


def save_results(output_file, correlated_pairs, metadata=None):
    """
    Save the final results of the correlation computation.
    
    Args:
        output_file (str): Path to the output file.
        correlated_pairs (list): List of correlated feature pairs.
        metadata (dict, optional): Additional metadata to save.
        
    Returns:
        str: Path to the saved output file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine the file format based on the extension
    _, ext = os.path.splitext(output_file)
    
    if ext.lower() == '.json':
        # Save as JSON
        results = {
            "metadata": metadata or {},
            "correlated_pairs": [
                {"feature_i": int(i), "feature_j": int(j), "correlation": float(corr)}
                for i, j, corr in correlated_pairs
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif ext.lower() in ['.h5', '.hdf5']:
        # Save as HDF5
        with h5py.File(output_file, 'w') as f:
            # Save metadata
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
            
            # Convert correlated pairs to a structured array
            if correlated_pairs:
                dtype = np.dtype([
                    ('feature_i', np.int32),
                    ('feature_j', np.int32),
                    ('correlation', np.float32)
                ])
                pairs_array = np.array(
                    [(i, j, corr) for i, j, corr in correlated_pairs],
                    dtype=dtype
                )
                f.create_dataset('correlated_pairs', data=pairs_array)
    
    elif ext.lower() == '.npy':
        # Save as NumPy array
        pairs_array = np.array(correlated_pairs)
        np.save(output_file, pairs_array)
    
    else:
        # Default to CSV
        pairs_array = np.array(correlated_pairs)
        np.savetxt(output_file, pairs_array, delimiter=',', 
                   header='feature_i,feature_j,correlation', comments='')
    
    print(f"Results saved to {output_file}")
    return output_file 