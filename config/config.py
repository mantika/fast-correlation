import argparse
import os
import json


class Config:
    """
    Configuration class for the correlation computation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the configuration with default values,
        and override them with the provided keyword arguments.
        """
        # Input/output
        self.input_file = kwargs.get('input_file', None)
        self.output_file = kwargs.get('output_file', None)
        self.dataset_key = kwargs.get('dataset_key', 'data')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.resume_from = kwargs.get('resume_from', None)
        
        # Computation parameters
        self.batch_size = kwargs.get('batch_size', 500)
        self.threshold = kwargs.get('threshold', 0.9)
        self.use_gpu = kwargs.get('use_gpu', True)
        self.gpu_ids = kwargs.get('gpu_ids', None)
        self.pruning_strategy = kwargs.get('pruning_strategy', 'first')
        
        # Checkpoint parameters
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 5)
        
        # Debug and logging
        self.verbose = kwargs.get('verbose', True)
        self.debug = kwargs.get('debug', False)
    
    @classmethod
    def from_args(cls, args):
        """
        Create a configuration from command-line arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments.
            
        Returns:
            Config: Configuration object.
        """
        # Convert namespace to dictionary
        config_dict = vars(args)
        
        # Create configuration object
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_file):
        """
        Create a configuration from a JSON file.
        
        Args:
            json_file (str): Path to the JSON configuration file.
            
        Returns:
            Config: Configuration object.
        """
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_json(self, json_file=None):
        """
        Save the configuration to a JSON file.
        
        Args:
            json_file (str, optional): Path to the output JSON file.
                If None, return the JSON string.
                
        Returns:
            str or None: JSON string if json_file is None, else None.
        """
        config_dict = self.__dict__
        
        if json_file is None:
            return json.dumps(config_dict, indent=2)
        
        with open(json_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __str__(self):
        """
        String representation of the configuration.
        
        Returns:
            str: String representation.
        """
        return f"Config({self.to_json()})"


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Fast correlation computation for large-scale tabular datasets.'
    )
    
    # Input/output arguments
    parser.add_argument('--input', dest='input_file', type=str, required=True,
                      help='Path to the input HDF5 file')
    parser.add_argument('--output', dest='output_file', type=str, required=True,
                      help='Path to the output file (CSV, JSON, HDF5, or NPY)')
    parser.add_argument('--dataset-key', dest='dataset_key', type=str, default='data',
                      help='Key for the dataset in the HDF5 file (default: data)')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--resume-from', dest='resume_from', type=str, default=None,
                      help='Path to a checkpoint file to resume from')
    
    # Computation parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=500,
                      help='Number of features in each batch (default: 500)')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.9,
                      help='Correlation threshold for feature pruning (default: 0.9)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                      help='Disable GPU acceleration')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('--gpus', dest='gpu_ids', type=str, default=None,
                      help='Comma-separated list of GPU IDs to use (default: all available)')
    parser.add_argument('--pruning-strategy', dest='pruning_strategy', type=str, default='first',
                      choices=['first', 'random'],
                      help='Strategy for selecting which feature to keep (default: first)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', type=int, default=5,
                      help='Number of batches between checkpoints (default: 5)')
    
    # Debug and logging
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                      help='Disable verbose output')
    parser.set_defaults(verbose=True)
    parser.add_argument('--debug', dest='debug', action='store_true',
                      help='Enable debug mode')
    parser.set_defaults(debug=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process GPU IDs
    if args.gpu_ids is not None:
        args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    
    return args 