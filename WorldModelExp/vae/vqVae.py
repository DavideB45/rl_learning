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

	def __init__(self, codebook_size:int, code_depth:int, latent_dim:int, commitment_cost:float, device:torch.device, ema_mode:bool=False):
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
		self.ema_mode = ema_mode

		self.quantizer = VectorQuantizer(codebook_size, code_depth, commitment_cost, ema=ema_mode)
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
			nn.Conv2d(3, 3, 3, 1, 1),                  # final conv layer
			nn.Sigmoid()
		)
		self.to(device)

	def param_count(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad) + sum(p.numel() for p in self.quantizer.parameters() if p.requires_grad)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Encodes input x into latent representation before quantization.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, 3, 64, 64)
		Returns:
			torch.Tensor: Latent representation of shape (batch, code_depth, latent_dim, latent_dim)
		"""
		return self.encoder(x)
	
	def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
	
	def encode_probabilities(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Encodes the latent representation z into codebook index probabilities.
		Args:
			z (torch.Tensor): Latent tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			torch.Tensor: Probabilities of shape (batch, codebook_size, latent_dim, latent_dim)
		"""
		return self.quantizer.get_index_probabilities(z)
	
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
			loss (torch.Tensor): Commitment loss + closeness loss.
			codebook_indices (torch.Tensor): Indices of the codebook vectors used.
		"""
		z = self.encode(x)
		loss, quantized, codebook_indices = self.quantize(z)
		recon = self.decode(quantized)
		return recon, loss, codebook_indices

	def reconstruction_loss(self, x, recon_x):
		return F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
	
	def train_epoch(self, loader:DataLoader, optim:torch.optim.Optimizer, reg:float) -> dict:
		'''
		Trains the VQVAE for one epoch.
		Args:
			loader (DataLoader): DataLoader for training data.
			optim (torch.optim.Optimizer): Optimizer for training.
			reg (float): Regularization strength (not used here).
		Returns:
			avg_loss (dict): Average loss over the epoch.
		'''
		losses = {
			"total_loss": 0.0,
			"recon_loss": 0.0,
			"commit_loss": 0.0,
			"codes_usage": 0.0
		}
		used_codes = set()
		for data in loader:
			data = data.to(self.device)
			optim.zero_grad()
			recon_batch, emb_loss, indexes = self(data)
			rec_loss = self.reconstruction_loss(data, recon_batch)
			loss = rec_loss + emb_loss
			loss.backward()
			optim.step()
			used_codes.update(indexes.view(-1).cpu().numpy().tolist())
			losses["total_loss"] += loss.item()
			losses["recon_loss"] += rec_loss.item()
			losses["commit_loss"] += emb_loss.item()
		for key in losses:
			losses[key] /= len(loader)
		losses["codes_usage"] = len(used_codes) / self.codebook_size
		return losses
	
	def eval_epoch(self, loader, reg):
		'''
		Evaluates the VQVAE for one epoch.
		Args:
			loader (DataLoader): DataLoader for validation data.
			reg (float): Regularization strength (not used here).
		Returns:
			avg_loss (dict): Average loss over the epoch.
		'''
		losses = {
			"total_loss": 0.0,
			"recon_loss": 0.0,
			"commit_loss": 0.0,
			"codes_usage": 0.0
		}
		used_codes = set()
		with torch.no_grad():
			for data in loader:
				data = data.to(self.device)
				recon_batch, emb_loss, indexes = self(data)
				rec_loss = self.reconstruction_loss(data, recon_batch)
				loss = rec_loss + emb_loss
				losses["total_loss"] += loss.item()
				losses["recon_loss"] += rec_loss.item()
				losses["commit_loss"] += emb_loss.item()
				used_codes.update(indexes.view(-1).cpu().numpy().tolist())
			for key in losses:
				losses[key] /= len(loader)
		losses["codes_usage"] = len(used_codes) / self.codebook_size
		return losses
	


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