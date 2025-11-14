import torch.nn as nn
from torch.nn.functional import leaky_relu

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