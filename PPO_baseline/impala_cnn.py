import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
	"""
	A single residual block with two 3x3 convolutions and skip connection.
	"""

	def __init__(self, channels: int):
		super().__init__()
		self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
		self.relu = nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		inputs = x
		out = self.relu(x)
		out = self.conv0(out)
		out = self.relu(out)
		out = self.conv1(out)
		return out + inputs


class ConvSequence(nn.Module):
	"""
	A convolutional stage: 3x3 Conv -> 3x3 MaxPool (stride 2) -> 2 ResidualBlocks.
	"""

	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.res_block0 = ResidualBlock(out_channels)
		self.res_block1 = ResidualBlock(out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = self.max_pool(x)
		x = self.res_block0(x)
		x = self.res_block1(x)
		return x


class ImpalaCNN(BaseFeaturesExtractor):
	"""
	IMPALA CNN feature extractor as described in Espeholt et al. (2018)
	and widely used in visual RL benchmarks (Procgen, Dreamer).

	Architecture:
	- Sequence of ConvSequence stages with residual blocks and max-pooling
	- Flatten + ReLU
	- Linear projection to `features_dim` latent embedding
	"""

	def __init__(
		self,
		observation_space: spaces.Box,
		features_dim: int = 256,
		depths: tuple = (16, 32, 32),
	):
		super().__init__(observation_space, features_dim)
		in_channels = observation_space.shape[0]

		layers = []
		for out_channels in depths:
			layers.append(ConvSequence(in_channels, out_channels))
			in_channels = out_channels

		layers.append(nn.Flatten())
		layers.append(nn.ReLU())
		self.network = nn.Sequential(*layers)

		# Compute the flattened feature dimension dynamically
		with torch.no_grad():
			dummy_input = torch.zeros(1, *observation_space.shape)
			n_flatten = self.network(dummy_input).shape[1]

		self.linear = nn.Sequential(
			nn.Linear(n_flatten, features_dim),
			nn.ReLU(),
		)

	def forward(self, observations: torch.Tensor) -> torch.Tensor:
		return self.linear(self.network(observations))
