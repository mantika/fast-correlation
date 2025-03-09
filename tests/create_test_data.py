#!/usr/bin/env python3

import os
import h5py
import numpy as np
import argparse


def create_test_dataset(output_file, n_samples=1000, n_features=5000, n_correlated_groups=50, 
                      correlation_strength=0.9, seed=42):
    """
    Create a test dataset with correlated features.
    
    Args:
        output_file (str): Path to the output HDF5 file.
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        n_correlated_groups (int): Number of groups of correlated features.
        correlation_strength (float): Strength of the correlation between features.
        seed (int): Random seed.
    """
    print(f"Creating test dataset with {n_samples} samples and {n_features} features")
    print(f"Creating {n_correlated_groups} groups of correlated features")
    
    # Set random seed
    np.random.seed(seed)
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate some parameters
    features_per_group = 5  # Number of features in each correlated group
    total_correlated_features = n_correlated_groups * features_per_group
    n_independent_features = n_features - total_correlated_features
    
    # Generate independent random features
    print(f"Generating {n_independent_features} independent features...")
    independent_features = np.random.randn(n_samples, n_independent_features)
    
    # Initialize the full feature matrix
    data = np.zeros((n_samples, n_features))
    data[:, :n_independent_features] = independent_features
    
    # Store ground truth correlated pairs
    correlated_pairs = []
    
    # Generate correlated feature groups
    print(f"Generating {n_correlated_groups} groups of correlated features...")
    for i in range(n_correlated_groups):
        # Generate a base feature
        base_feature = np.random.randn(n_samples)
        
        # Calculate the start index for this group in the feature matrix
        start_idx = n_independent_features + i * features_per_group
        
        # Set the base feature
        data[:, start_idx] = base_feature
        
        # Generate correlated features
        for j in range(1, features_per_group):
            # Create a feature correlated with the base feature
            noise = np.random.randn(n_samples) * np.sqrt(1 - correlation_strength**2)
            correlated_feature = correlation_strength * base_feature + noise
            
            # Add the correlated feature
            feat_idx = start_idx + j
            data[:, feat_idx] = correlated_feature
            
            # Add this pair to the ground truth
            correlated_pairs.append((start_idx, feat_idx, correlation_strength))
    
    # Save the dataset to HDF5
    print(f"Saving dataset to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Save the data
        f.create_dataset('data', data=data)
        
        # Add metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['n_features'] = n_features
        f.attrs['n_correlated_groups'] = n_correlated_groups
        f.attrs['features_per_group'] = features_per_group
        f.attrs['correlation_strength'] = correlation_strength
        
        # Save ground truth correlated pairs
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
            f.create_dataset('correlated_pairs_truth', data=pairs_array)
    
    print(f"Created {len(correlated_pairs)} correlated feature pairs")
    print(f"Dataset saved to {output_file}")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description='Create a test dataset with correlated features.'
    )
    
    parser.add_argument('--output', type=str, required=True,
                      help='Path to the output HDF5 file')
    parser.add_argument('--n-samples', type=int, default=1000,
                      help='Number of samples (default: 1000)')
    parser.add_argument('--n-features', type=int, default=5000,
                      help='Number of features (default: 5000)')
    parser.add_argument('--n-correlated-groups', type=int, default=50,
                      help='Number of groups of correlated features (default: 50)')
    parser.add_argument('--correlation-strength', type=float, default=0.9,
                      help='Strength of the correlation between features (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_test_dataset(
        args.output,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_correlated_groups=args.n_correlated_groups,
        correlation_strength=args.correlation_strength,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 