import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from vae.vqVae import VQVAE

class LSTMQuantized(nn.Module):
	def __init__(self, quantizer:VQVAE, device:torch.device, action_dim:int, hidden_dim:int=512):
		super(LSTMQuantized, self).__init__()
		self.w_h = quantizer.latent_dim
		self.d = quantizer.code_depth
		self.latent_dim = self.w_h*self.w_h*self.d
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.quantizer = quantizer

		self.rep_fc = nn.Sequential(
			nn.Linear(self.latent_dim, hidden_dim),
			nn.LeakyReLU()
		)
		self.act_fc = nn.Sequential(
			nn.Linear(action_dim, self.latent_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(self.latent_dim),
			nn.Linear(self.latent_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim)
		)
		self.merge_fc = nn.Sequential(
			nn.Linear(self.hidden_dim*2, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU()
		)
		self.lstm = LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
		self.out_fc = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, self.latent_dim)
		)

		self.device = device
		self.to(device)

	def flatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Takes as input the unflattened output (possibly quantized) of the vq-vae \
		and flattens it based on the VQVAE model used.
		Each consecutive 'Depth' elements of the output compose a vector 
		TODO: possibly add a possibility to make this work for normal vae doing noop 
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		Returns:
			torch.Tensor: the flattened input (Batch, Seq_len, Width*Height*Depth)
		'''
		input = input.permute(0, 1, 3, 4, 2).contiguous() # (B, S, W, H, D)
		input = input.view(input.size(0), input.size(1), -1) # (B, S, W*H*D)
		return input
	
	def unflatten_rep(self, input:torch.Tensor, s:int) -> torch.Tensor:
		'''
		Takes as input a flat tensor of the LSTM \
		and unflatten it based on the VQVAE model used.
		Each consecutive Depth elements of the output compose a vector 
		TODO: possibly add a possibility to make this work for normal vae doing noop 
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Seq_len, Width*Height*Depth)
			s (int): The lenght of the original sequence
		Returns:
			torch.Tensor: the flattened input (Batch, Seq_len, Depth, Width, Height)
		'''
		b = input.size(0)
		w = self.w_h
		h = w
		d = self.d

		input = input.view(b, s, w, h, d)
		input = input.permute(0, 1, 4, 2, 3).contiguous()
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
		return F.mse_loss(y, x, reduction='sum') / x.size(0)


	def forward(self, input:torch.Tensor, action:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, Seq_len, Depth, Width, Height)
			action (torch.Tensor): a tensor representing the robot action
			h (tuple): initial hidden state (optional)
		Returns:
			torch.Tensor: the predicted sequence
			torch.Tensor: the predicted sequence quantized
			tuple[torch.Tensor, torch.Tensor]: the hidden state of the LSTM
		'''
		input = self.flatten_rep(input.detach())
		new_rep = self.rep_fc(input.detach())
		action = self.act_fc(action)
		output = torch.cat([new_rep, action], dim=-1)
		skip_output = self.merge_fc(output) #(B, Seq_len, Hidden_dim)

		if h is None:
			h = (torch.zeros(1, input.size(0), self.hidden_dim).to(input.device),
			     torch.zeros(1, input.size(0), self.hidden_dim).to(input.device))
		output, h = self.lstm(skip_output, h)

		output = output + skip_output #(B, Seq_len, Hidden_dim)
		output = self.out_fc(output) #(B, Seq_len, Height*Width*Depth)
		output = output + input.detach()
		output = self.unflatten_rep(output, input.size(1)) # (B, Seq_len, Depth, Height, Width)
		
		_, q_output, _ = self.quantizer.quantize(output.view(-1, self.d, self.w_h, self.w_h))
		q_output = q_output.view(input.size(0), input.size(1), self.d, self.w_h, self.w_h)
		
		return output, q_output, h
	
	def generate_sequence(self, input:torch.Tensor, action:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture
		
		Args:
			input (torc.Tensor): Input tensor shape (Batch, 1, Depth, Width, Height)
			action (torch.Tensor): a tensor representing the robot action (Batch, Seq_len, Act_size)
			h (tuple): initial hidden state (optional)
		Returns:
			torch.Tensor: the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height)
			tuple[torch.Tensor, torch.Tensor]: the hidden state of the LSTM
		'''
		batch, len, _ = action.shape
		device = input.device
		preds = []

		if h is None:
			h = (torch.zeros(1, batch, self.hidden_dim).to(device),
			     torch.zeros(1, batch, self.hidden_dim).to(device))

		x = input.detach()

		for t in range(len):
			x = self.flatten_rep(x)
			rep = self.rep_fc(x)

			a = action[:, t:t+1, :]
			a = self.act_fc(a)
			rep = torch.cat([rep, a], dim=-1)
			skip = self.merge_fc(rep)
			rep, h = self.lstm(skip, h)
			rep = rep + skip

			rep = self.out_fc(rep)
			rep = rep + x
			rep = self.unflatten_rep(rep, 1)
			_, rep, _ = self.quantizer.quantize(rep.view(-1, self.d, self.w_h, self.w_h))
			rep = rep.view(batch, 1, self.d, self.w_h, self.w_h)
			preds.append(rep)
			x = rep

		preds = torch.cat(preds, dim=1)
		return preds, h
	
	def train_epoch(self, loader:DataLoader, optim:Optimizer) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			output, q_output, _ = self.forward(input=latent[:, :-1, :, :, :], action=action)
			loss = F.mse_loss(latent[:, 1:, :, :, :], output, reduction='mean')# / output.size(0)
			q_loss = F.mse_loss(latent[:, 1:, :, :, :], q_output, reduction='mean')
			loss.backward()
			optim.step()
			total_loss += loss.item()
			total_q_loss += q_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader)
		}
	
	def eval_epoch(self, loader:DataLoader) -> dict:
		self.eval()
		total_loss = 0
		total_q_loss = 0
		with torch.no_grad():
			for batch in loader:
				latent = batch['latent'].to(self.device)
				action = batch['action'].to(self.device)
				output, q_output, _ = self.forward(input=latent[:, :-1, :, :, :], action=action)
				loss = F.mse_loss(latent[:, 1:, :, :, :], output, reduction='mean')# / output.size(0)
				q_loss = F.mse_loss(latent[:, 1:, :, :, :], q_output, reduction='mean')
				total_loss += loss.item()
				total_q_loss += q_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader)
		}

