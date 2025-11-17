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
	def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float):
		super().__init__()
		self.codebook_size = codebook_size
		self.embedding_dim = embedding_dim
		self.commitment_cost = commitment_cost
		self.embedding = nn.Embedding(codebook_size, embedding_dim)
		self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
		input_shape = x.shape
		flat_x = x.view(-1, 1, self.embedding_dim)  # (B*H*W, 1, D)
		distances = (flat_x - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)  # (B*H*W, K)
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)
		quantized = self.embedding(encoding_indices).view(input_shape)  # (B, H, W, D)
		
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		if self.training:
			quantized = x + (quantized - x).detach()
		
		return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.reshape(input_shape[0], -1)