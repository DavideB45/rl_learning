import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu

from typing import Tuple

class ResidualBlock(nn.Module):
	"""
	A residual block with optional downsampling.
	"""
	def __init__(self, in_channels, out_channels, downsample=False):
		"""
		Initializes the ResidualBlock.
		Args:
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			downsample (bool): Whether to downsample the input.
		"""
		super().__init__()
		stride = 2 if downsample else 1
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.skip = nn.Sequential()
		if downsample or in_channels != out_channels:
			self.skip = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		out = leaky_relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.skip(x)
		return leaky_relu(out)
	
class ResidualBlockUp(nn.Module):
	"""
	A residual block with optional upsampling.
	"""
	def __init__(self, in_channels, out_channels, upsample=False):
		"""
		Initializes the ResidualBlockUp.
		Args:
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			upsample (bool): Whether to upsample the input.
		"""
		super().__init__()
		stride = 2 if upsample else 1
		self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride, 1, output_padding=1 if upsample else 0)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.skip = nn.Sequential()
		if upsample or in_channels != out_channels:
			self.skip = nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=1 if upsample else 0),
				nn.BatchNorm2d(out_channels)
			)
		
	def forward(self, x):
		out = leaky_relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.skip(x)
		return leaky_relu(out)
	
class VectorQuantizer(nn.Module):
	"""
	A vector quantizer layer for VQ-VAE. 
	"""
	def __init__(self, 
			  codebook_size: int, embedding_dim: int, 
			  commitment_cost: float, 
			  ema: bool = False, gamma: float = 0.99, epsilon: float = 1e-5):
		super().__init__()
		self.codebook_size = codebook_size
		self.embedding_dim = embedding_dim
		self.commitment_cost = commitment_cost
		self.embedding = nn.Embedding(codebook_size, embedding_dim)
		self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
		self.ema = ema
		if ema:
			self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
			self.register_buffer("ema_weights", self.embedding.weight.detach().clone())
			self.gamma = gamma
			self.epsilon = epsilon

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Forward pass of the vector quantizer.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			commitment_loss (torch.Tensor): Commitment loss.
			quantized (torch.Tensor): Quantized tensor of shape (batch, code_depth, latent_dim, latent_dim)
			codebook_indices (torch.Tensor): Indices of the codebook vectors used.
		"""
		x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
		input_shape = x.shape
		flat_x = x.view(-1, 1, self.embedding_dim)  # (B*H*W, 1, D)
		distances = (flat_x - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)  # (B*H*W, K)
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)
		quantized = self.embedding(encoding_indices).view(input_shape)  # (B, H, W, D)
		
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		if self.ema:
			loss = e_latent_loss * self.commitment_cost
		else:
			q_latent_loss = F.mse_loss(quantized, x.detach())
			loss = q_latent_loss + self.commitment_cost * e_latent_loss

		if self.training:
			if self.ema:
				# Cluster size update
				one_hot = F.one_hot(encoding_indices.squeeze(), self.codebook_size).float()
				cluster_size = one_hot.sum(0) # in each position, how many vectors assigned to each codebook entry
				self.ema_cluster_size = self.gamma * self.ema_cluster_size.to(self.embedding.weight.device) + (1 - self.gamma) * cluster_size # N_i*gamma + (1-gamma)*n_i
				# Laplace smoothing (avoid empty clusters)
				n = self.ema_cluster_size.sum()
				self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
				# Weights update
				dw = one_hot.t() @ flat_x.squeeze(1).detach()
				self.ema_weights = self.gamma * self.ema_weights.to(self.embedding.weight.device) + (1 - self.gamma) * dw
				
				self.embedding.weight.data = self.ema_weights / self.ema_cluster_size.unsqueeze(1)
			quantized = x + (quantized - x).detach()
		
		return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.reshape(input_shape[0], -1)
	
	def quantize_fixed_space(self, z:torch.Tensor) -> torch.Tensor:
		"""
		Quantize the input z using the codebook without updating the embeddings.
		But allows gradient flow using straight-through estimator.

		Args:
			z (torch.Tensor): Input tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			quantized (torch.Tensor): Quantized tensor of shape (batch, code_depth, latent_dim, latent_dim)
		"""
		z = z.permute(0, 2, 3, 1).contiguous()
		input_shape = z.shape
		flat_z = z.view(-1, 1, self.embedding_dim)
		distances = (flat_z - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		quantized = self.embedding(encoding_indices).view(input_shape)
		quantized = z + (quantized - z).detach()
		return quantized.permute(0, 3, 1, 2).contiguous()

	def get_index_probabilities(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Computes the probabilities of each codebook index for the input x.
		Args:
			x (torch.Tensor): Input tensor of shape (batch, code_depth, latent_dim, latent_dim)
		Returns:
			torch.Tensor: Probabilities of shape (batch, codebook_size, latent_dim, latent_dim)
		"""

		x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
		input_shape = x.shape
		flat_x = x.view(-1, 1, self.embedding_dim)  # (B*H*W, 1, D)
		distances = (flat_x - self.embedding.weight.unsqueeze(0).detach()).pow(2).mean(2)  # (B*H*W, K)
		# Convert distances to probabilities
		probabilities = F.softmax(-distances, dim=1)  # (B*H*W, K)
		return probabilities.view(input_shape[0], input_shape[1], input_shape[2], self.codebook_size).permute(0, 3, 1, 2).contiguous()  # (B, K, H, W)
	
	def vec_from_prob(self, x: torch.Tensor) -> torch.Tensor:
		'''
		Get the latent space vector corresponding to the higher number in the input 
		
		:param x: A vector (output from a sigmoid or a one hot encoded) of size code_depth
		:type x: torch.Tensor
		:return: the vector corresponding to the argmax
		:rtype: torch.Tensor
		'''
		indices = torch.argmax(x, dim=1)
		return self.embedding(indices)
	
	def onehot_from_vec(self, x:torch.Tensor) -> torch.Tensor:
		'''
		Convert a latent vector to its one-hot encoded representation based on closest codebook vector.
		
		:param x: Input tensor of shape (batch, code_depth, latent_dim, latent_dim)
		:type x: torch.Tensor
		:return: One-hot encoded tensor of shape (batch, codebook_size, latent_dim, latent_dim)
		:rtype: torch.Tensor
		'''
		x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
		input_shape = x.shape
		flat_x = x.view(-1, 1, self.embedding_dim)  # (B*H*W, 1, D)
		distances = (flat_x - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)  # (B*H*W, K)
		encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
		one_hot = F.one_hot(encoding_indices, self.codebook_size).float()  # (B*H*W, K)
		return one_hot.view(input_shape[0], input_shape[1], input_shape[2], self.codebook_size).permute(0, 3, 1, 2).contiguous()  # (B, K, H, W)
