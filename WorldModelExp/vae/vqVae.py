import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.abstractVAE import AbstractVAE
from vae.blocks import ResidualBlock, VectorQuantizer, ResidualBlockUp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


class VQVAE(AbstractVAE):
	"""
	An implementation of the VQVAE model.
	Take as input images of size 3x64x64.
	TODO: add tricks for using the full codebook
	"""

	def __init__(self, codebook_size:int, code_depth:int, latent_dim:int, commitment_cost:float, device):
		"""
		Initialize the VQVAE model.
		Teh embedding will be of size (code_depth, latent_dim, latent_dim)
		Args:
			codebook_size (int): Number of codebook vector that exist. (K)
			code_depth (int): The dimension of each codebook vector. (D)
			latent_dim (int): The dimension of the latent representation. (8 or 4, if something else is required explicitly define the encoder/decoder after initialization)
			commitment_cost (float): The committment cost for the quantizzation loss.
			device (torch.device): The device to run the model on.
		"""
		
		super().__init__(latent_dim, device)
		self.codebook_size = codebook_size
		self.code_depth = code_depth
		self.commitment_cost = commitment_cost

		self.quantizer = VectorQuantizer(codebook_size, code_depth, commitment_cost)

		self.encoder = nn.Sequential(
			ResidualBlock(3, 32, downsample=True),  # 64x64 -> 32x32
			ResidualBlock(32, 64, downsample=True),  # 32x32 -> 16x16
			ResidualBlock(64, 128, downsample=True), # 16x16 -> 8x8
			ResidualBlock(128, code_depth, downsample=(latent_dim==4)) # 8x8 -> latent_dimxlatent_dim
		)

		self.decoder = nn.Sequential(
			ResidualBlockUp(code_depth, 128, upsample=(latent_dim==4)), # latent_dimxlatent_dim -> 8x8
			ResidualBlockUp(128, 64, upsample=True),   # 8x8 -> 16x16
			ResidualBlockUp(64, 32, upsample=True),    # 16x16 -> 32x32
			ResidualBlockUp(32, 3, upsample=True),     # 32x32 -> 64x64
		)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Encodes input x into latent representation before quantization.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, 3, 64, 64)
		Returns:
			torch.Tensor: Latent representation of shape (batch, code_depth, latent_dim, latent_dim)
		"""
		return self.encoder(x)
	
	def embed(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Embeds the latent representation z using the vector quantizer.
		Args:
			z (torch.Tensor): Latent tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			commitment_loss (torch.Tensor): Commitment loss.
			quantized (torch.Tensor): Quantized tensor of shape (batch, code_depth, latent_dim, latent_dim)
			codebook_indices (torch.Tensor): Indices of the codebook vectors used.
		"""
		return self.quantizer(z)
	
	def decode(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Decodes latent z to reconstruction space (e.g. image or feature space).
		To work the input z should be composed of codebook vectors.
		Args:
			z (torch.Tensor): Latent tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			torch.Tensor: Reconstructed tensor of shape (batch, 3, 64, 64)
		"""
		return self.decoder(z)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Standard forward pass returning reconstruction, commitment_loss, codebook_indices.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, 3, 64, 64)
		Returns:
			recon (torch.Tensor): Reconstructed tensor of shape (batch, 3, 64, 64)
			commitment_loss (torch.Tensor): Commitment loss.
			codebook_indices (torch.Tensor): Indices of the codebook vectors used.
		"""
		z = self.encode(x)
		commitment_loss, quantized, codebook_indices = self.embed(z)
		recon = self.decode(quantized)
		return recon, commitment_loss, codebook_indices

	def eval_epoch(self, loader, reg):
		return super().eval_epoch(loader, reg)
	
	def train_epoch(self, loader, optim, reg):
		return super().train_epoch(loader, optim, reg)
	
	def reconstruction_loss(self, x, recon_x):
		return super().reconstruction_loss(x, recon_x)


if __name__ == "__main__":
	model = VQVAE(codebook_size=512, code_depth=64, latent_dim=4, commitment_cost=0.25, device=torch.device("cpu"))
	print(model)
	print("Number of parameters:", model.count_parameters())

	# Test forward pass
	x = torch.randn(4, 3, 64, 64)  # Batch
	recon, commit_loss, codebook_indices = model(x)
	print("Input shape:", x.shape)
	print("Reconstructed shape:", recon.shape)
	print("Commitment loss shape:", commit_loss.shape)
	print("Codebook indices shape:", codebook_indices.shape)