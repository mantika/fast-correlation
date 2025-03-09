import torch
import numpy as np


def compute_pearson_correlation(X, Y=None):
    """
    Compute the Pearson correlation coefficient between features in X,
    or between features in X and Y if Y is provided.
    
    Args:
        X (torch.Tensor): Tensor of shape [n_samples, n_features_X]
        Y (torch.Tensor, optional): Tensor of shape [n_samples, n_features_Y]
            If None, correlations will be computed between features in X.
            
    Returns:
        torch.Tensor: Correlation matrix of shape [n_features_X, n_features_Y]
            or [n_features_X, n_features_X] if Y is None.
    """
    # Center the data (subtract mean)
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    if Y is None:
        Y_centered = X_centered
    else:
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute standard deviations
    X_std = torch.sqrt(torch.sum(X_centered ** 2, dim=0) / (X.shape[0] - 1))
    Y_std = torch.sqrt(torch.sum(Y_centered ** 2, dim=0) / (X.shape[0] - 1))
    
    # Compute covariance matrix
    n_samples = X.shape[0]
    cov = torch.matmul(X_centered.T, Y_centered) / (n_samples - 1)  # Use n-1 for unbiased estimate
    
    # Compute correlation matrix
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    denominator = torch.outer(X_std, Y_std) + epsilon
    correlation = cov / denominator
    
    # Ensure values are in the range [-1, 1]
    correlation = torch.clamp(correlation, -1.0, 1.0)
    
    # Set diagonal elements to exactly 1.0 if X and Y are the same
    if Y is None:
        correlation.fill_diagonal_(1.0)
    
    return correlation


def compute_batch_correlation(batch_features, all_features, active_indices=None):
    """
    Compute correlations between a batch of features and all active features.
    
    Args:
        batch_features (torch.Tensor): Tensor of shape [n_samples, batch_size]
        all_features (torch.Tensor): Tensor of shape [n_samples, n_features]
        active_indices (list, optional): List of active feature indices.
            If None, all features are considered active.
            
    Returns:
        torch.Tensor: Correlation matrix of shape [batch_size, n_active_features]
        list: List of active feature indices
    """
    if active_indices is None:
        active_indices = list(range(all_features.shape[1]))
    
    # Select only active features
    active_features = all_features[:, active_indices]
    
    # Compute correlation between batch features and active features
    correlation_matrix = compute_pearson_correlation(batch_features, active_features)
    
    # Manually set correlations for identical features
    for i, batch_idx in enumerate(range(batch_features.shape[1])):
        for j, active_idx in enumerate(active_indices):
            if batch_idx == active_idx:
                correlation_matrix[i, j] = 1.0
    
    return correlation_matrix, active_indices


def find_correlated_pairs(correlation_matrix, batch_indices, active_indices, threshold=0.9):
    """
    Find pairs of features that are highly correlated.
    
    Args:
        correlation_matrix (torch.Tensor): Correlation matrix of shape [batch_size, n_active_features]
        batch_indices (list): List of feature indices in the batch
        active_indices (list): List of active feature indices
        threshold (float): Correlation threshold for feature pruning
        
    Returns:
        list: List of correlated feature pairs [(i, j, correlation), ...] where
            i is from batch_indices and j is from active_indices
    """
    # Find indices where abs(correlation) exceeds threshold
    mask = torch.abs(correlation_matrix) > threshold
    
    # Convert to CPU and numpy for easier processing
    mask_np = mask.cpu().numpy()
    corr_np = correlation_matrix.cpu().numpy()
    
    correlated_pairs = []
    # For each batch feature
    for i in range(mask_np.shape[0]):
        batch_idx = batch_indices[i]
        # For each active feature
        for j in range(mask_np.shape[1]):
            active_idx = active_indices[j]
            # Skip if this is a self-correlation (same feature index)
            if batch_idx == active_idx:
                continue
            # Skip if this is a correlation with another batch feature
            # that will be handled in a different iteration
            if active_idx in batch_indices and active_idx > batch_idx:
                continue
            # If correlation exceeds threshold
            if mask_np[i, j]:
                # Skip if correlation is exactly 1.0 (likely a self-correlation)
                if abs(corr_np[i, j]) == 1.0:
                    continue
                # Add the pair with its correlation value
                correlated_pairs.append((batch_idx, active_idx, corr_np[i, j]))
    
    return correlated_pairs


def update_active_indices(active_indices, correlated_pairs, strategy='first'):
    """
    Update the list of active feature indices by removing redundant features.
    
    Args:
        active_indices (list): Current list of active feature indices
        correlated_pairs (list): List of correlated feature pairs [(i, j, corr), ...]
        strategy (str): Strategy for selecting which feature to keep
            'first': Keep the feature with the lower index
            'random': Randomly select which feature to keep
            
    Returns:
        list: Updated list of active feature indices
    """
    indices_to_remove = set()
    
    for i, j, corr in correlated_pairs:
        # If both features are still active
        if i in active_indices and j in active_indices:
            if strategy == 'first':
                # Keep the feature with the lower index
                to_remove = max(i, j)
            elif strategy == 'random':
                # Randomly select which feature to remove
                to_remove = i if np.random.random() < 0.5 else j
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            indices_to_remove.add(to_remove)
    
    # Create new active indices list excluding the ones to remove
    updated_indices = [idx for idx in active_indices if idx not in indices_to_remove]
    
    return updated_indices, indices_to_remove 