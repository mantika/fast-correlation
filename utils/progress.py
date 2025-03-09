import time
import torch
from tqdm import tqdm


class ProgressTracker:
    """
    A class for tracking progress and performance metrics during correlation computation.
    """
    
    def __init__(self, total_features, batch_size, checkpoint_freq=5):
        """
        Initialize the progress tracker.
        
        Args:
            total_features (int): Total number of features in the dataset.
            batch_size (int): Number of features in each batch.
            checkpoint_freq (int): Frequency of checkpoints in number of batches.
        """
        self.total_features = total_features
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq
        
        # Calculate number of batches
        self.total_batches = (total_features + batch_size - 1) // batch_size
        
        # Initialize progress bar
        self.pbar = tqdm(total=self.total_batches, desc="Processing batches")
        
        # Initialize metrics
        self.start_time = time.time()
        self.batch_times = []
        self.pruned_features = []
        self.active_features_history = []
        self.correlated_pairs_count = 0
        
        # Initialize batch index
        self.current_batch = 0
    
    def start_batch(self):
        """
        Start timing a new batch.
        
        Returns:
            float: Start time of the batch.
        """
        self.batch_start_time = time.time()
        return self.batch_start_time
    
    def end_batch(self, n_active, n_pruned, n_pairs, gpu_memory=None):
        """
        End timing of the current batch and update metrics.
        
        Args:
            n_active (int): Number of active features after pruning.
            n_pruned (int): Number of features pruned in this batch.
            n_pairs (int): Number of correlated pairs found in this batch.
            gpu_memory (float, optional): GPU memory usage in GB.
        """
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        # Update metrics
        self.current_batch += 1
        self.pruned_features.append(n_pruned)
        self.active_features_history.append(n_active)
        self.correlated_pairs_count += n_pairs
        
        # Update progress bar
        elapsed = time.time() - self.start_time
        avg_time_per_batch = elapsed / self.current_batch
        eta = avg_time_per_batch * (self.total_batches - self.current_batch)
        
        memory_str = f", GPU: {gpu_memory:.2f} GB" if gpu_memory is not None else ""
        
        self.pbar.set_postfix({
            'active': n_active,
            'pruned': sum(self.pruned_features),
            'pairs': self.correlated_pairs_count,
            'batch_time': f"{batch_time:.2f}s",
            'eta': f"{eta/60:.1f}m{memory_str}"
        })
        self.pbar.update(1)
        
        # Check if it's time for a checkpoint
        should_checkpoint = (self.current_batch % self.checkpoint_freq == 0 or
                           self.current_batch == self.total_batches)
        
        return should_checkpoint
    
    def get_gpu_memory_usage(self, device=None):
        """
        Get the current GPU memory usage.
        
        Args:
            device (torch.device, optional): Device to check memory usage for.
                If None, check all devices.
                
        Returns:
            float: GPU memory usage in GB, or None if not using GPU.
        """
        if not torch.cuda.is_available():
            return None
        
        if device is None:
            # Get maximum memory usage across all devices
            memory_allocated = max(
                torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
            )
        else:
            # Get memory usage for specific device
            device_idx = device.index if device.type == 'cuda' else 0
            memory_allocated = torch.cuda.memory_allocated(device_idx)
        
        # Convert to GB
        return memory_allocated / (1024 ** 3)
    
    def get_elapsed_time(self):
        """
        Get the total elapsed time since the start.
        
        Returns:
            float: Elapsed time in seconds.
        """
        return time.time() - self.start_time
    
    def get_summary(self):
        """
        Get a summary of the progress and performance metrics.
        
        Returns:
            dict: Dictionary containing summary metrics.
        """
        elapsed = self.get_elapsed_time()
        
        return {
            "total_features": self.total_features,
            "total_batches": self.total_batches,
            "completed_batches": self.current_batch,
            "active_features": self.active_features_history[-1] if self.active_features_history else self.total_features,
            "pruned_features": sum(self.pruned_features),
            "correlated_pairs": self.correlated_pairs_count,
            "elapsed_time": elapsed,
            "avg_time_per_batch": elapsed / max(1, self.current_batch),
            "progress_percent": 100 * self.current_batch / self.total_batches
        }
    
    def close(self):
        """
        Close the progress bar and finalize metrics.
        """
        self.pbar.close()
        
        # Print summary
        summary = self.get_summary()
        print(f"\nCompleted {summary['completed_batches']}/{summary['total_batches']} batches "
              f"({summary['progress_percent']:.1f}%)")
        print(f"Found {summary['correlated_pairs']} correlated pairs")
        print(f"Pruned {summary['pruned_features']} features "
              f"({100 * summary['pruned_features'] / self.total_features:.1f}%)")
        print(f"Total elapsed time: {summary['elapsed_time']/60:.1f} minutes") 