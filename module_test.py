try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Failed to import torch: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"Failed to import numpy: {e}")

try:
    import h5py
    print(f"h5py version: {h5py.__version__}")
except ImportError as e:
    print(f"Failed to import h5py: {e}")

try:
    import tqdm
    print(f"tqdm version: {tqdm.__version__}")
except ImportError as e:
    print(f"Failed to import tqdm: {e}")

# Print Python path for debugging
import sys
print("\nPython path:")
for path in sys.path:
    print(path) 