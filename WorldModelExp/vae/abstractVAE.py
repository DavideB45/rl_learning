from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import torch

import torch.nn as nn


class AbstractVAE(nn.Module, ABC):
	"""
	Minimal abstract base class for a Variational Autoencoder.
	Subclasses must implement encode, decode, and reconstruction_loss.
	"""

	def __init__(self, latent_dim: int, device: torch.device = torch.device("cpu")):
		super().__init__()
		self.latent_dim = latent_dim
		self.device = device

	@abstractmethod
	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Encode input x into latent Gaussian parameters.
		Returns:
			mu: (batch, latent_dim)
			logvar: (batch, latent_dim)
		"""
		raise NotImplementedError

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, force_sampling: bool = False) -> torch.Tensor:
		"""
		Reparameterization trick: sample z ~ N(mu, sigma^2) using mu and logvar.
		In eval mode returns mu (deterministic).
		"""
		if not self.training and not force_sampling:
			return mu
		std = (0.5 * logvar).exp()
		eps = torch.randn_like(std)
		return mu + eps * std

	@abstractmethod
	def decode(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Decode latent z to reconstruction space (e.g. image or feature space).
		"""
		raise NotImplementedError

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Standard forward pass returning reconstruction, mu, logvar.
		"""
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar

	def sample(self, num_samples: int) -> torch.Tensor:
		"""
		Sample from the prior (standard normal) and decode to observation space.
		"""
		z = torch.randn(num_samples, self.latent_dim, device=self.device)
		return self.decode(z)

	def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		"""
		Returns average KL divergence per batch between q(z|x) = N(mu, var) and p(z)=N(0,I).
		"""
		# sum over latent dims, mean over batch
		kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		return kl.mean()

	@abstractmethod
	def reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		"""
		Reconstruction loss between recon_x and x (e.g. BCE or MSE).
		Should return a scalar (mean over batch).
		"""
		raise NotImplementedError

	def loss_function(
		self,
		recon_x: torch.Tensor,
		x: torch.Tensor,
		mu: torch.Tensor,
		logvar: torch.Tensor,
		beta: float = 1.0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Combined loss: reconstruction + beta * KL.
		Returns: total_loss, diagnostics dict
		"""
		rec = self.reconstruction_loss(recon_x, x)
		kl = self.kl_divergence(mu, logvar)
		total = rec + beta * kl
		return total, {"reconstruction": rec.detach(), "kl": kl.detach()}