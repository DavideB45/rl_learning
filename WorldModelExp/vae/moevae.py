import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import torch
from torch import nn
from typing import Tuple
from vae.myVae import CVAE


class MOEVAE(nn.Module):
	"""
	A simple implementation of a mixture of experts VAE.
	Each expert is a CVAE model for now.
	Also, only two experts are implemented.
	"""

	def __init__(self, latent_dim:int, device:torch.device, learn_gating:bool=True):
		super(MOEVAE, self).__init__()
		self.expert1 = CVAE(latent_dim=latent_dim, device=device)
		self.expert2 = CVAE(latent_dim=latent_dim, device=device)
		self.latent_dim = latent_dim
		self.device = device
		
		self.gating_network = None
		if learn_gating:
			self.gating_network = nn.Sequential(
				nn.Linear(3 * 64 * 64, 128),
				nn.ReLU(),
				nn.Linear(128, 2),
				nn.Softmax(dim=1)
			)
		else:
			self.gating_network = lambda x: torch.tensor([[0.5, 0.5]] * x.size(0)).to(device)
		self.to(device)

	def forward(self, x1:torch.Tensor, x2:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Forward pass through the MOE VAE.
		Args:
			x1 (torch.Tensor): Input tensor for expert 1 of shape (batch, 3, 64, 64)
			x2 (torch.Tensor): Input tensor for expert 2 of shape (batch, 3, 64, 64)
		Returns:
			recon_x (torch.Tensor): Reconstructed tensor of shape (batch, 3, 64, 64)
			mu (torch.Tensor): Mean of the latent Gaussian (batch, latent_dim)
			logvar (torch.Tensor): Log-variance of the latent Gaussian (batch, latent_dim)
		"""
		batch_size = x1.size(0)
		x1_flat = x1.view(batch_size, -1)
		gating_weights = self.gating_network(x1_flat)  # (batch, 2)

		recon1, mu1, logvar1 = self.expert1.forward(x1)
		recon2, mu2, logvar2 = self.expert2.forward(x2)

		mu = (gating_weights[:, 0].unsqueeze(1) * mu1 +
				gating_weights[:, 1].unsqueeze(1) * mu2)
		logvar = (gating_weights[:, 0].unsqueeze(1) * logvar1 +
					gating_weights[:, 1].unsqueeze(1) * logvar2)

		return recon1, recon2, mu, logvar
	
	def encode(self, x1:torch.Tensor, x2:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Encodes input x into latent Gaussian parameters.
		Args:
			x1 (torch.Tensor): Input tensor for expert 1 of shape (batch, 3, 64, 64)
			x2 (torch.Tensor): Input tensor for expert 2 of shape (batch, 3, 64, 64)
		Returns:
			mu (torch.Tensor): Mean of the latent Gaussian (batch, latent_dim)
			logvar (torch.Tensor): Log-variance of the latent Gaussian (batch, latent_dim
		"""
		batch_size = x1.size(0)
		x1_flat = x1.view(batch_size, -1)
		gating_weights = self.gating_network(x1_flat)  # (batch, 2)

		mu1, logvar1 = self.expert1.encode(x1)
		mu2, logvar2 = self.expert2.encode(x2)

		mu = (gating_weights[:, 0].unsqueeze(1) * mu1 +
				gating_weights[:, 1].unsqueeze(1) * mu2)
		logvar = (gating_weights[:, 0].unsqueeze(1) * logvar1 +
					gating_weights[:, 1].unsqueeze(1) * logvar2)

		return mu, logvar
	
	def encode_expert1(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.expert1.encode(x)
	
	def encode_expert2(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.expert2.encode(x)
	
	def decode_expert1(self, z:torch.Tensor) -> torch.Tensor:
		return self.expert1.decode(z)
	
	def decode_expert2(self, z:torch.Tensor) -> torch.Tensor:
		return self.expert2.decode(z)
	
	def kl_divergence(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
		"""
		Computes the KL divergence between the learned latent distribution and the prior.
		Args:
			mu (torch.Tensor): Mean of the latent Gaussian (batch, latent_dim)
			logvar (torch.Tensor): Log-variance of the latent Gaussian (batch, latent_dim)
		Returns:
			torch.Tensor: KL divergence loss.
		"""
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		kl_loss /= mu.size(0)  # average over batch
		return kl_loss
	
	def reconstruction_loss(self, x:torch.Tensor, recon_x:torch.Tensor) -> torch.Tensor:
		"""
		Computes the reconstruction loss between input x and its reconstruction.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, 3, 64, 64)
			recon_x (torch.Tensor): Reconstructed tensor of shape (batch, 3, 64, 64)
		Returns:
			torch.Tensor: Reconstruction loss.
		"""
		return nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
	
	def compute_loss(self, x1:torch.Tensor, x2:torch.Tensor, reg_strength:float=1.0) -> torch.Tensor:
		"""
		Computes the total loss (reconstruction + KL divergence).
		Args:
			x1 (torch.Tensor): Input tensor for expert 1 of shape (batch, 3, 64, 64)
			x2 (torch.Tensor): Input tensor for expert 2 of shape (batch, 3, 64, 64)
			reg_strength (float): Weight for the KL divergence term.
		Returns:
			torch.Tensor: Total loss.
		"""
		recon1, recon2, mu, logvar = self.forward(x1, x2)
		recon_loss1 = self.reconstruction_loss(x1, recon1)
		recon_loss2 = self.reconstruction_loss(x2, recon2)
		kl_loss = self.kl_divergence(mu, logvar)
		total_loss = recon_loss1 + recon_loss2 + reg_strength * kl_loss
		return total_loss
		
	def train_epoch(self, loader:torch.utils.data.DataLoader, optim:torch.optim.Optimizer, reg:float) -> dict:
		avg_loss = 0.0
		for data in loader:
			x1, x2 = data  # assuming data is a tuple of (x1, x2)
			x1 = x1.to(self.device)
			x2 = x2.to(self.device)
			optim.zero_grad()
			loss = self.compute_loss(x1, x2, reg)
			loss.backward()
			optim.step()
			avg_loss += loss.item()
		avg_loss = avg_loss / len(loader)
		return {"avg_loss": avg_loss}
	
	def eval_epoch(self, loader:torch.utils.data.DataLoader, reg:float) -> dict:
		avg_loss = 0.0
		with torch.no_grad():
			for data in loader:
				x1, x2 = data  # assuming data is a tuple of (x1, x2)
				x1 = x1.to(self.device)
				x2 = x2.to(self.device)
				loss = self.compute_loss(x1, x2, reg)
				avg_loss += loss.item()
		avg_loss = avg_loss / len(loader)
		return {"avg_loss": avg_loss}