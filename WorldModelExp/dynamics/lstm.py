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

	def forward(self, input, action, h=None):
		input = self.rep_fc(input)
		action = self.act_fc(action)
		output = torch.cat(input, action, dim=-1)
		output = self.merge_fc(output)
		output, (h,c) = self.lstm(output, h)
		# create a 4*4*channel tensor
		# encode it
		# reflatten it 