import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.abstractVAE import AbstractVAE
from vae.resBlock import ResidualBlock

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple


class CVAE(AbstractVAE):
	"""
	The variational autoencoder model used in the final project
	for the exam of Introduction to Robotics:
	TODO: add a description of the architecture
	"""

	def __init__(self, latent_dim: int, device: torch.device = torch.device("cpu")):
		"""
		Initializes the VAE model.
		"""

		super().__init__(latent_dim, device)

		# ---------- Encoder ----------
		self.encoder = nn.Sequential(
			ResidualBlock(3, 32, downsample=True),   # 64x64 -> 32x32
			ResidualBlock(32, 64, downsample=True),  # 32x32 -> 16x16
			ResidualBlock(64, 128, downsample=True), # 16x16 -> 8x8
			ResidualBlock(128, 256, downsample=True) # 8x8 -> 4x4
		)
		self.enc_fc = nn.Linear(256 * 4 * 4, 512)
		self.fc_mu = nn.Linear(512, latent_dim)
		self.fc_logvar = nn.Linear(512, latent_dim)

		# ---------- Decoder ----------
		self.dec_fc = nn.Linear(latent_dim, 512)
		self.dec_expand = nn.Linear(512, 256 * 4 * 4)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32x32 -> 64x64
		)
		
	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Encodes input x into latent Gaussian parameters.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, 3, 64, 64)
		Returns:
			mu (torch.Tensor): Mean of the latent Gaussian (batch, latent_dim)
			logvar (torch.Tensor): Log-variance of the latent Gaussian (batch, latent_dim)
		"""
		h = self.encoder(x)
		h = h.view(h.size(0), -1)
		h = F.leaky_relu(self.enc_fc(h))
		return self.fc_mu(h), self.fc_logvar(h)
	
	def decode(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Decodes latent z to reconstruction space (image).
		Args:
			z (torch.Tensor): Latent tensor of shape (batch, latent_dim)
		Returns:
			recon (torch.Tensor): Reconstructed images (batch, 3, 64, 64)
		"""
		h = F.leaky_relu(self.dec_fc(z))
		h = F.leaky_relu(self.dec_expand(h)).view(-1, 256, 4, 4)
		recon = self.decoder(h)
		return torch.sigmoid(recon)
	
	def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
		"""
		Computes the reconstruction loss between input x and its reconstruction.
		Args:
			x (torch.Tensor): Original input tensor (batch, 3, 64, 64)
			recon (torch.Tensor): Reconstructed tensor (batch, 3, 64, 64)
		Returns:
			loss (torch.Tensor): Scalar tensor representing the reconstruction loss
		"""
		return F.mse_loss(recon, x, reduction='sum') / x.size(0)