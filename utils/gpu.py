import torch


def get_available_gpus():
    """
    Get the list of available GPUs.
    
    Returns:
        list: List of available GPU IDs.
    """
    if not torch.cuda.is_available():
        return []
    
    n_gpus = torch.cuda.device_count()
    return list(range(n_gpus))


def get_device(gpu_id=None):
    """
    Get a PyTorch device object for the specified GPU ID.
    
    Args:
        gpu_id (int, optional): GPU ID to use. If None, use CPU.
            If -1, use the first available GPU.
            
    Returns:
        torch.device: PyTorch device object.
    """
    if gpu_id is None:
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        return torch.device('cpu')
    
    n_gpus = torch.cuda.device_count()
    
    if gpu_id == -1:
        # Use the first available GPU
        if n_gpus > 0:
            return torch.device('cuda:0')
        else:
            print("No GPUs available. Using CPU instead.")
            return torch.device('cpu')
    
    if gpu_id >= n_gpus:
        print(f"GPU {gpu_id} is not available. Using CPU instead.")
        return torch.device('cpu')
    
    return torch.device(f'cuda:{gpu_id}')


def setup_multi_gpu(gpu_ids=None):
    """
    Set up devices for multi-GPU processing.
    
    Args:
        gpu_ids (list, optional): List of GPU IDs to use.
            If None, use all available GPUs.
            
    Returns:
        list: List of torch.device objects for each GPU.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        return [torch.device('cpu')]
    
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No GPUs available. Using CPU instead.")
        return [torch.device('cpu')]
    
    if gpu_ids is None:
        # Use all available GPUs
        gpu_ids = available_gpus
    else:
        # Filter out invalid GPU IDs
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpus]
        
        if not gpu_ids:
            print("No valid GPUs in the provided list. Using CPU instead.")
            return [torch.device('cpu')]
    
    return [torch.device(f'cuda:{gpu_id}') for gpu_id in gpu_ids]


def print_gpu_info():
    """
    Print information about available GPUs.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")
    
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        
        print(f"GPU {i}: {gpu_name}")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Memory reserved: {memory_reserved:.2f} GB") 