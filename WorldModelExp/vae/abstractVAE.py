from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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
	def reconstruction_loss(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
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
		regularization_strength: float = 1.0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Combined loss: reconstruction + regularization_strength * KL.
		Returns: total_loss, diagnostics dict
		"""
		rec = self.reconstruction_loss(x, recon_x)
		kl = self.kl_divergence(mu, logvar)
		total = rec + regularization_strength * kl
		return total, {"reconstruction": rec.detach(), "kl": kl.detach()}
	
	@abstractmethod
	def train_epoch(self, loader:DataLoader, optim:torch.optim.Optimizer, reg:float) -> dict:
		avg_loss = 0.0
		for data in loader:
			data = data.to(self.device)
			optim.zero_grad()
			recon_batch, mu, logvar = self(data)
			loss, v = self.loss_function(recon_batch, data, mu, logvar, reg)
			loss.backward()
			train_loss += loss.item()
			optim.step()
			avg_loss += loss.item()
		avg_loss = avg_loss / len(loader)
		return {avg_loss}
	
	@abstractmethod
	def eval_epoch(self, loader:DataLoader, reg:float) -> dict:
		avg_loss = 0.0
		with torch.no_grad:
			for data in loader:
				data = data.to(self.device)
				recon_batch, mu, logvar = self(data)
				loss, v = self.loss_function(recon_batch, data, mu, logvar, reg)
				avg_loss += loss.item()
		return {avg_loss / len(loader)}
	
	def count_parameters(self) -> int:
		"""
		Returns the total number of trainable parameters in the model.
		"""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	
def trainVAE(vae: AbstractVAE,
			 train_loader: DataLoader,
			 val_loader: DataLoader,
			 num_epochs: int,
			 learning_rate: float,
			 regularization_strength: float = 1.0) -> Tuple[AbstractVAE, dict]:
	"""
	Trains the VAE model.
	Args:
		vae (AbstractVAE): The VAE model to train.
		train_loader (DataLoader): DataLoader for training data.
		val_loader (DataLoader): DataLoader for validation data.
		num_epochs (int): Number of training epochs.
		learning_rate (float): Learning rate for the optimizer.
		regularization_strength (float): Weight for the KL divergence term.
	Returns:
		AbstractVAE: The trained VAE model.
		dict: Training and validation loss history.
	"""
	device = vae.device
	optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
	loss_history = {
		'train_loss': {
			'kl': [],
			'reconstruction': [],
			'total': []
		}, 'val_loss': {
			'kl': [],
			'reconstruction': [],
			'total': []
		}
	}

	for _ in tqdm(range(num_epochs), desc='Training VAE'):
		vae.train()
		train_loss = vae.train_epoch(train_loader, optimizer, regularization_strength)
		loss_history['train_loss']['total'].append(train_loss['avg_loss'])

		vae.eval()
		val_loss = vae.eval_epoch(val_loader, regularization_strength)
		loss_history['val_loss']['total'].append(val_loss['avg_loss'])

	return vae, loss_history