# Fast Correlation

A high-performance Python tool for identifying correlated features in large-scale tabular datasets using PyTorch.

## Features

- Fast Pearson correlation computation using GPU acceleration
- Multi-GPU support for parallel processing
- Dynamic feature pruning to optimize computation
- Batch processing for memory efficiency
- HDF5 input file support
- Progress tracking and visualization

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- h5py for HDF5 file handling
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fast-correlation.git
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

Basic usage:

```bash
python main.py --input data.h5 --output correlated_features.json --threshold 0.9 --batch-size 500
```

With GPU selection:

```bash
python main.py --input data.h5 --output correlated_features.json --threshold 0.9 --batch-size 500 --gpus 0,1
```

## Parameters

- `--input`: Path to the input HDF5 file (required)
- `--output`: Path to save the output file (required)
- `--dataset-key`: Key for the dataset in the HDF5 file (default: 'data')
- `--threshold`: Correlation threshold for feature pruning (default: 0.9)
- `--batch-size`: Number of features to process in each batch (default: 500)
- `--gpus`: Comma-separated list of GPU IDs to use (default: all available)

## Project Structure

- `data_loading/`: HDF5 data handling
- `correlation/`: Core correlation computation logic
- `utils/`: Helper functions (progress tracking, feature pruning)
- `config/`: Configuration files
- `main.py`: Entry point for running the tool
- `tests/`: Unit tests

## License

MIT 