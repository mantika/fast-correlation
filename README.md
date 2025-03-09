# Fast Correlation

A high-performance Python tool for identifying correlated features in large-scale tabular datasets using PyTorch.

## Features

- Fast Pearson correlation computation using GPU acceleration
- Multi-GPU support for parallel processing
- Dynamic feature pruning to optimize computation
- Batch processing for memory efficiency
- HDF5 input file support
  - Single file loading
  - Multi-file loading with memory management
- Progress tracking and visualization

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- h5py for HDF5 file handling
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mantika/fast-correlation.git
cd fast-correlation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --input data.h5 --output correlated_features.json --threshold 0.9 --batch-size 500
```

### With GPU Selection

```bash
python main.py --input data.h5 --output correlated_features.json --threshold 0.9 --batch-size 500 --gpus 0,1
```

### Using Multiple HDF5 Files

For very large datasets split across multiple files:

```bash
python main.py --input-dir data_directory --file-pattern "*.h5" --output correlated_features.json --threshold 0.9 --batch-size 500
```

## Parameters

- `--input`: Path to the input HDF5 file
- `--input-dir`: Directory containing multiple HDF5 files (alternative to --input)
- `--file-pattern`: Pattern to match HDF5 files in the directory (default: "*.h5")
- `--output`: Path to save the output file (required)
- `--dataset-key`: Key for the dataset in the HDF5 file(s) (default: 'data')
- `--threshold`: Correlation threshold for feature pruning (default: 0.9)
- `--batch-size`: Number of features to process in each batch (default: 500)
- `--gpus`: Comma-separated list of GPU IDs to use (default: all available)
- `--max-files-in-memory`: Maximum number of files to load into memory at once (default: 2)

## Project Structure

- `data_loading/`: HDF5 data handling
- `correlation/`: Core correlation computation logic
- `utils/`: Helper functions (progress tracking, feature pruning)
- `config/`: Configuration files
- `main.py`: Entry point for running the tool
- `tests/`: Unit tests

## Data Loading

### Single File Loading

```python
from data_loading import HDF5DataLoader

# Initialize the data loader
loader = HDF5DataLoader(file_path='data.h5', dataset_key='data')

# Load specific features
feature_indices = [0, 5, 10, 15]
data = loader.load_features(feature_indices)

# Process in batches
batch_size = 100
for start_idx, end_idx in loader.get_feature_batches(batch_size):
    batch_data = loader.load_batch(start_idx, end_idx)
    # Process batch_data
```

### Multi-File Loading

```python
from data_loading import MultiFileHDF5DataLoader

# Initialize the multi-file data loader
loader = MultiFileHDF5DataLoader(
    directory_path='data_directory',
    file_pattern='*.h5',
    dataset_key='data',
    max_files_in_memory=2  # Control memory usage
)

# Load specific features across all files
feature_indices = [0, 5, 10, 15]
data = loader.load_features(feature_indices)

# Process in batches optimized for file boundaries
batch_size = 1000
for start_idx, end_idx in loader.get_sample_batches(batch_size):
    batch_data = loader.load_sample_range(start_idx, end_idx)
    # Process batch_data
```

## License

MIT 
