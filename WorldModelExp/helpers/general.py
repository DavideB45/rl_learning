from torch import device

def best_device() -> device:
	"""
	Returns the best available device (GPU if available, MPS if on Mac, else CPU).
	"""
	import torch
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif torch.backends.mps.is_available():
		return torch.device("mps")
	else:
		return torch.device("cpu")