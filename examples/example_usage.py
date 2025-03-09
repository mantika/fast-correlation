#!/usr/bin/env python3

import os
import sys
import time
import argparse
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loading import HDF5DataLoader
from correlation.pearson import compute_pearson_correlation
from utils.gpu import get_available_gpus, get_device, print_gpu_info
from tests.create_test_data import create_test_dataset


def create_demo_dataset(output_file):
    """
    Create a demo dataset for the example.
    
    Args:
        output_file (str): Path to the output HDF5 file.
    """
    # Create a small dataset for demonstration
    create_test_dataset(
        output_file,
        n_samples=1000,
        n_features=2000,
        n_correlated_groups=50,
        correlation_strength=0.9
    )


def run_example(input_file, output_file):
    """
    Run an example correlation computation.
    
    Args:
        input_file (str): Path to the input HDF5 file.
        output_file (str): Path to the output JSON file.
    """
    # Check for available GPUs
    available_gpus = get_available_gpus()
    if available_gpus:
        print(f"Found {len(available_gpus)} GPUs")
        device = get_device(0)  # Use the first GPU
    else:
        print("No GPUs found, using CPU")
        device = get_device(None)  # Use CPU
    
    # Print GPU information
    print_gpu_info()
    
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    loader = HDF5DataLoader(input_file, 'data')
    
    # Print dataset information
    print(f"Dataset shape: {loader.shape}")
    print(f"Number of samples: {loader.n_samples}")
    print(f"Number of features: {loader.n_features}")
    
    # For a small dataset, we can compute the full correlation matrix
    if loader.n_features <= 2000:
        print("\nComputing full correlation matrix...")
        start_time = time.time()
        
        # Load all data
        data = loader.load_all_data(device=device)
        
        # Compute correlation matrix
        corr_matrix = compute_pearson_correlation(data)
        
        # Find correlated feature pairs
        threshold = 0.8
        correlated_pairs = []
        
        print(f"Finding feature pairs with correlation above {threshold}...")
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                corr = corr_matrix[i, j].item()
                if abs(corr) >= threshold:
                    correlated_pairs.append({
                        "feature_i": int(i),
                        "feature_j": int(j),
                        "correlation": float(corr)
                    })
        
        elapsed_time = time.time() - start_time
        print(f"Found {len(correlated_pairs)} correlated feature pairs")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Save results to JSON
        results = {
            "metadata": {
                "n_samples": loader.n_samples,
                "n_features": loader.n_features,
                "threshold": threshold,
                "elapsed_time": elapsed_time
            },
            "correlated_pairs": correlated_pairs
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    else:
        print("\nDataset is too large for full correlation matrix computation.")
        print("For large datasets, use the main.py script with batch processing.")
        print("\nExample command:")
        print(f"python main.py --input {input_file} --output {output_file} " + 
              "--batch-size 500 --threshold 0.8")


def main():
    """
    Main entry point for the example script.
    """
    parser = argparse.ArgumentParser(
        description='Example usage of the fast correlation tool.'
    )
    
    parser.add_argument('--create-dataset', action='store_true',
                      help='Create a demo dataset')
    parser.add_argument('--input', type=str, default='demo_data.h5',
                      help='Path to the input HDF5 file (default: demo_data.h5)')
    parser.add_argument('--output', type=str, default='correlated_pairs.json',
                      help='Path to the output JSON file (default: correlated_pairs.json)')
    
    args = parser.parse_args()
    
    # Create demo dataset if requested
    if args.create_dataset:
        print(f"Creating demo dataset at {args.input}")
        create_demo_dataset(args.input)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        print("Use --create-dataset to create a demo dataset")
        return
    
    # Run the example
    run_example(args.input, args.output)


if __name__ == "__main__":
    main() 