import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Set the GPU device (e.g., GPU 0)
    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    # Use CPU if GPU is not available
    device = torch.device("cpu")
    print("GPU not available, using CPU")