# Fast Correlation - Project Architecture

This document provides an overview of the architecture and implementation details of the Fast Correlation project.

## Project Overview

Fast Correlation is a Python tool for identifying correlated features in large-scale tabular datasets using PyTorch. It leverages GPU acceleration, supports multiple GPUs, and implements dynamic feature pruning to optimize computation. The tool is designed to handle large datasets (50k+ features, 1M+ samples) efficiently.

## Directory Structure

The project is organized into the following modules:

- `data_loading/`: HDF5 data handling
  - `hdf5_loader.py`: Efficient loading of large HDF5 datasets
- `correlation/`: Core correlation computation logic
  - `pearson.py`: Pearson correlation computation and feature pruning
- `utils/`: Helper functions
  - `gpu.py`: GPU detection and management
  - `checkpoint.py`: Checkpoint and result saving/loading
  - `progress.py`: Progress tracking and metrics
- `config/`: Configuration management
  - `config.py`: Command-line arguments and configuration
- `tests/`: Unit and integration tests
  - `create_test_data.py`: Test dataset generation
  - `test_correlation.py`: Tests for correlation computation
  - `test_data_loader.py`: Tests for data loading
- `examples/`: Example usage scripts
  - `example_usage.py`: Simple demonstration of the tool
- `main.py`: Entry point for running the tool

## Key Components

### Data Loading (HDF5DataLoader)

The `HDF5DataLoader` class provides efficient loading of large datasets from HDF5 files. Key features:

- Lazy loading: Only loads metadata initially, not the entire dataset
- Batch loading: Loads data in batches to manage memory usage
- Feature selection: Can load specific subsets of features
- GPU support: Directly transfers data to GPU memory
- NaN handling: Replaces NaN values with zeros

### Correlation Computation

The correlation computation is implemented in the `correlation.pearson` module:

- `compute_pearson_correlation`: Computes Pearson correlation coefficient between features
- `compute_batch_correlation`: Computes correlations between a batch of features and all active features
- `find_correlated_pairs`: Identifies pairs of features with correlation above a threshold
- `update_active_indices`: Updates the list of active features by pruning correlated ones

### Dynamic Feature Pruning

To optimize computation for large datasets, the tool implements dynamic feature pruning:

1. Start with all features as "active"
2. Process features in batches
3. For each batch, compute correlations with all active features
4. Find pairs of features with correlation above the threshold
5. Remove redundant features (using either 'first' or 'random' strategy)
6. Continue with the reduced set of active features

This approach significantly reduces the number of correlations to compute as the process progresses.

### GPU Acceleration

The tool leverages PyTorch's GPU acceleration for matrix operations:

- GPU detection and selection
- Multi-GPU support for parallel processing
- GPU memory management
- Fallback to CPU when GPUs are not available

### Checkpointing

The tool provides checkpoint functionality to handle interruptions and resume processing:

- Regular checkpoints during batch processing
- Saves the current state (active indices, correlated pairs)
- Can resume from a checkpoint file

### Progress Tracking

The `ProgressTracker` class provides detailed progress information during processing:

- Batch progress and estimated time remaining
- Number of active features and pruned features
- Number of correlated pairs found
- GPU memory usage

## Implementation Details

### Batch Processing Strategy

Due to the quadratic complexity of computing all pairwise correlations, the tool processes features in batches:

1. Select a batch of features (e.g., 500 features)
2. Load the batch features and all active features to GPU memory
3. Compute correlations between the batch and active features
4. Find correlated pairs and update active features
5. Proceed to the next batch

### Correlation Computation Optimization

Pearson correlation is computed efficiently using PyTorch tensor operations:

```python
# Center the data (subtract mean)
X_centered = X - X.mean(dim=0, keepdim=True)
Y_centered = Y - Y.mean(dim=0, keepdim=True)

# Compute standard deviations
X_std = torch.sqrt(torch.sum(X_centered ** 2, dim=0))
Y_std = torch.sqrt(torch.sum(Y_centered ** 2, dim=0))

# Compute covariance matrix
n_samples = X.shape[0]
cov = torch.matmul(X_centered.T, Y_centered) / n_samples

# Compute correlation matrix
denominator = torch.outer(X_std, Y_std) + epsilon
correlation = cov / denominator
```

### Feature Pruning Strategy

When correlated features are identified, the tool needs to decide which feature to keep and which to prune. Two strategies are supported:

1. `first`: Keep the feature with the lower index (deterministic)
2. `random`: Randomly select which feature to keep (non-deterministic)

## Performance Considerations

The tool includes several optimizations for performance:

- Batch processing to manage memory usage
- Dynamic feature pruning to reduce computation
- GPU acceleration for matrix operations
- Efficient data loading from HDF5 files
- Progress tracking and checkpointing for long-running processes

## Scalability

The tool is designed to scale with the size of the dataset:

- For small datasets (<2000 features), it can compute the full correlation matrix directly
- For large datasets, it uses batch processing and feature pruning
- For very large datasets, it supports multi-GPU processing

## Future Improvements

Potential future improvements include:

- Support for other correlation metrics (Spearman, Kendall)
- Distributed computing across multiple machines using PyTorch distributed
- Mixed-precision computation for additional speed
- Direct visualization of correlation matrices or feature clusters
- Incremental computation for streaming data 