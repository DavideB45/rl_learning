import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from vae.vqVae import VQVAE

class FNN(nn.Module):
	def __init__(self, quantizer:VQVAE, device:torch.device, action_dim:int, history_len=2):
		super(FNN, self).__init__()
		self.w_h = quantizer.latent_dim
		self.d = quantizer.code_depth
		self.latent_dim = self.w_h*self.w_h*self.d
		self.action_dim = action_dim
		self.quantizer = quantizer
		self.history = history_len

		# input is composed by 2 (for now) consecutive cose
		in_dim = self.latent_dim*history_len + action_dim 
		hidden_block = nn.Sequential(
			nn.Linear(in_dim*4, in_dim*4),
			nn.LeakyReLU(),
			nn.BatchNorm1d(in_dim*4)
		)

		self.net = nn.Sequential(
			nn.Linear(in_dim, in_dim*4),
			nn.LeakyReLU(),
			nn.BatchNorm1d(in_dim*4),
			hidden_block,
			hidden_block,
			hidden_block,
			hidden_block,
			nn.Linear(in_dim*4, self.latent_dim),
		)

		self.device = device
		self.to(device)

	def flatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Takes as input the unflattened output (possibly quantized) of the vq-vae \
		and flattens it based on the VQVAE model used.
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, history_len, Depth, Width, Height)
		Returns:
			torch.Tensor: the flattened input (Batch, history_len*Width*Height*Depth)
		'''

		input = input.permute(0, 1, 3, 4, 2).contiguous() # (B, S, W, H, D)
		input = input.view(input.size(0), -1) # (B, S*W*H*D)
		return input
	
	def unflatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Unflatten the predicted sate
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Width*Height*Depth)
		Returns:
			torch.Tensor: the flattened input (Batch, Depth, Width, Height)
		'''
		b = input.size(0)
		w = self.w_h
		h = w
		d = self.d

		input = input.view(b, w, h, d)
		input = input.permute(0, 3, 1, 2).contiguous()
		return input
	
	def mse_loss(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
		'''
		Docstring for loss
		
		Args:
			x (torch.Tensor): ground truth
			y (torch.Tensor): the prediction
		Returns:
			dict: the loss
		'''
		return F.mse_loss(y, x, reduction='mean')


	def forward(self, input:torch.Tensor, action:torch.Tensor, stop_grad:bool=True) -> tuple[torch.Tensor, torch.Tensor]:
		'''
		Do a single forward pass (meaning only one step will be taken)
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, history_len, Depth, Width, Height)
			action (torch.Tensor): a tensor representing the robot action (Batch, 1, Act_size)
		Returns:
			torch.Tensor: the predicted next state
			torch.Tensor: the predicted next state quantized
		'''
		
		if(stop_grad):
			input = input.detach()

		input = self.flatten_rep(input)
		out = torch.cat([input, action], dim=1)
		
		out = self.net(out)
		out = out + input[:, self.latent_dim*(self.history -1):].detach()

		out = self.unflatten_rep(out)
		out_q = self.quantizer.quantizer.quantize_fixed_space(out)
		
		return out, out_q
	
	def ar_forward(self, input:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		'''
		Do the forward pass in an autoregressive way using until consuming all the actions, no teacher forcing will be used
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, 2, Depth, Width, Height)
			action (torch.Tensor): a tensor representing the robot action (Batch, Seq_len, Act_size)
			h (tuple): initial hidden state (optional)
		Returns:
			torch.Tensor: the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height)
			tuple[torch.Tensor, torch.Tensor]: the hidden state of the LSTM
		'''
		raise NotImplementedError()
		batch, len, _ = action.shape
		device = input.device
		preds = []

		x = input.detach()

		for t in range(len):
			out, q_out, h = self.forward(x, action[:, t:t+1, :], h)
			print(f'Distance between consecutive = {F.mse_loss(x, out, reduction="mean")}')
			preds.append(q_out)
			x = q_out
			
		preds = torch.cat(preds, dim=1)
		return preds, h

	def train_epoch(self, loader:DataLoader, optim:Optimizer, autoregressive:bool) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			optim.zero_grad()
			if autoregressive:
				raise NotImplementedError()
				output, q_output, _ = self.generate_sequence(latent[:, 0:1, :, :, :], action=action)
			else:
				output, q_output = self.forward(input=latent[:, 0:self.history, :, :, :], action=action[:, self.history-1, :])
				loss = F.mse_loss(latent[:, self.history, :, :, :], output, reduction='mean')
				q_loss = F.mse_loss(latent[:, self.history, :, :, :], q_output, reduction='mean')
			loss.backward()
			optim.step()
			total_loss += loss.item()
			total_q_loss += q_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
		}
	
	def eval_epoch(self, loader:DataLoader, autoregressive:bool) -> dict:
		self.eval()
		total_loss = 0
		total_q_loss = 0
		with torch.no_grad():
			for batch in loader:
				latent = batch['latent'].to(self.device)
				action = batch['action'].to(self.device)
				if autoregressive:
					raise NotImplementedError()
					output, q_output, _ = self.generate_sequence(latent[:, 0:1, :, :, :], action=action)
				else:
					output, q_output = self.forward(input=latent[:, 0:self.history, :, :, :], action=action[:, self.history-1, :])
					loss = F.mse_loss(latent[:, self.history, :, :, :], output, reduction='mean')
					q_loss = F.mse_loss(latent[:, self.history, :, :, :], q_output, reduction='mean')
				total_loss += loss.item()
				total_q_loss += q_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
		}

