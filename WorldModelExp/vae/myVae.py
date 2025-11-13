from abstractVAE import AbstractVAE
import torch
import torch.nn as nn
from typing import Tuple, Optional


class VAE(AbstractVAE):
	"""
	The variational autoencoder model used in the final project
	for the exam of Introduction to Robotics:
	TODO: add a description of the architecture
	"""

	def __init__(self, latent_dim: int, device: torch.device = torch.device("cpu")):
		super().__init__(latent_dim, device)

		