import torch

def best_device():
    '''
    Returns the best available device (GPU or MPS if available, else CPU)
    '''
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")