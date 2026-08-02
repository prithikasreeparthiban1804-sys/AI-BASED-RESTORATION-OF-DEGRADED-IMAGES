import torch

print("PyTorch Version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
    print("Using CPU")