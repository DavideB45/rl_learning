import torch
import torch.nn as nn
from torch.nn import LSTM

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from vae.vqVae import VQVAE

class LSTMQuantized(nn.Module):
	def __init__(self, latent_dim:int=32, action_dim:int=6, hidden_dim:int=512, quantizer:VQVAE=None):
		super(LSTMQuantized, self).__init__()
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.quantizer = quantizer

		self.rep_fc = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.ReLU()
		)
		self.act_fc = nn.Sequential(
			nn.Linear(action_dim, latent_dim),
			nn.ReLU()
		)
		self.merge_fc = nn.Sequential(
			nn.Linear(latent_dim*2, hidden_dim),
			nn.ReLU(),
			nn.BatchNorm1d(hidden_dim)
		)
		self.lstm = LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)

		# TODO: sommare nel mezzo
		self.out_fc = nn.Sequential(
			nn.BatchNorm1d(hidden_dim),
			nn.Linear(hidden_dim, latent_dim)
		)

	def flatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Takes as input the unflattened output (possibly quantized) of the vq-vae \
		and flattens it based on the VQVAE model used.
		Each consecutive Depth elements of the output compose a vector 
		TODO: possibly add a possibility to make this work for normal vae doing noop 
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Depth, Width, Height)
		Returns:
			torch.Tensor: the flattened input (Batch, Width*Height*Depth)
		'''
		input = input.permute(0, 2, 3, 1).contiguous() # (B, W, H, D)
		input = input.view(input.size(0), -1) # (B, W*H*D)
		return input
	
	def unflatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Takes as input a flat tensor of the LSTM \
		and unflatten it based on the VQVAE model used.
		Each consecutive Depth elements of the output compose a vector 
		TODO: possibly add a possibility to make this work for normal vae doing noop 
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Width*Height*Depth)
		Returns:
			torch.Tensor: the flattened input (Batch, Depth, Width, Height)
		'''
		

	def forward(self, input, action, h=None):
		# flatten input (should talk the VQ-VAE language)
		input = self.rep_fc(input)
		action = self.act_fc(action)
		output = torch.cat(input, action, dim=-1)
		skip_output = self.merge_fc(output)
		output, (h,c) = self.lstm(skip_output, h)
		output = output + skip_output #(B, Depth*Height*Width)


		# create a 4*4*channel tensor
		# encode it
		# reflatten it 