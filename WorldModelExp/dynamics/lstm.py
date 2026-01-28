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
from helpers.metrics import weighted_mse, change_mse

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
			nn.LayerNorm(hidden_dim + action_dim),
			nn.Linear(hidden_dim + action_dim, hidden_dim),
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
		Takes as input a flat tensor of the LSTM 
		and unflatten it based on the VQVAE model used.
		Each consecutive Depth elements of the output compose a vector 
		
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

	def forward(self, input:torch.Tensor, action:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture
		
		:param input: Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action
		:type action: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) | the hidden state of the LSTM
		:rtype: tuple[Tensor, Tensor, tuple[Tensor, Tensor]]
		'''
		input = self.flatten_rep(input)
		new_rep = self.rep_fc(input)
		action2 = self.act_fc(action)
		output = torch.cat([new_rep, action2], dim=-1)
		skip_output = self.merge_fc(output) #(B, Seq_len, Hidden_dim)

		if h is None:
			h = (torch.zeros(1, input.size(0), self.hidden_dim).to(input.device),
				 torch.zeros(1, input.size(0), self.hidden_dim).to(input.device))
		output, h = self.lstm(skip_output, h)

		output = output + skip_output #(B, Seq_len, Hidden_dim)
		#print(output.shape)
		#print(action.shape)
		output = torch.cat([output, action], dim=-1)
		#print(output.shape)
		#exit()
		output = self.out_fc(output) #(B, Seq_len, Height*Width*Depth)
		output = output + input
		output = self.unflatten_rep(output, input.size(1)) # (B, Seq_len, Depth, Height, Width)
		
		_, q_output, _ = self.quantizer.quantize(output.view(-1, self.d, self.w_h, self.w_h))
		q_output = q_output.view(input.size(0), input.size(1), self.d, self.w_h, self.w_h)
		
		return output, q_output, h
	
	def ar_forward(self, input:torch.Tensor, action:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture in an autoregressive fashion
		
		:param input: Input tensor shape (Batch, 1, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action (Batch, Seq_len, Act_size)
		:type action: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) | the hidden state of the LSTM
		:rtype: tuple[Tensor, Tensor, tuple[Tensor, Tensor]]
		'''
		batch, len, _ = action.shape
		preds_q = []
		preds = []

		if h is None:
			h = (torch.zeros(1, batch, self.hidden_dim).to(self.device),
				 torch.zeros(1, batch, self.hidden_dim).to(self.device))
		x = input.detach()
		for t in range(len):
			out, q_out, h = self.forward(x, action[:, t:t+1, :], h)
			preds_q.append(q_out)
			preds.append(out)
			x = q_out
			
		preds_q = torch.cat(preds_q, dim=1)
		preds = torch.cat(preds, dim=1)
		return preds, preds_q, h
	
	def compute_classification_target(self, target:torch.Tensor) -> torch.Tensor:
		'''
		Takes as input the unflattened target and encodes it into a one hot encoding vector

		Args:
			target (torc.Tensor): Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		Returns:
			torch.Tensor: the flattened input (Batch, Seq_len, Width, Height, Classes)
		'''
		b = target.size(0)
		s = target.size(1)
		w = self.quantizer.latent_dim
		h = w
		c = self.quantizer.codebook_size
		d = self.quantizer.code_depth # depth

		target = target.contiguous().view(b*s, d, w, h) # (B*S, D, W, H)
		#target = self.quantizer.quantizer.onehot_from_vec(target) # (B*S, C, W, H)
		target = self.quantizer.quantizer.get_index_probabilities(target)
		target = target.view(b, s, c, w, h).contiguous() # (B, S, C, W, H)
		target = target.permute(0, 1, 3, 4, 2) # (B, S, W, H, C)
		return target
	
	def train_epoch(self, loader:DataLoader, optim:Optimizer) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			optim.zero_grad()
			output, q_output, _ = self.forward(input=latent[:, :-1, :, :, :], action=action)
			loss = F.mse_loss(latent[:, 1:, :, :, :], output, reduction='mean')
			q_loss = F.mse_loss(latent[:, 1:, :, :, :], q_output, reduction='mean')
			loss.backward()
			optim.step()
			total_loss += loss.item()
			total_q_loss += q_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
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
			'qmse': total_q_loss/len(loader),
		}

	def train_rwm_style(self, loader:DataLoader, optim:Optimizer, init_len:int=3, err_decay:float=0.9) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		accuracy = 0.0
		for batch in loader:
			latent = batch['latent'].to(self.device).detach()
			action = batch['action'].to(self.device).detach()
			optim.zero_grad()
			_, _, h = self.forward(latent[:, 0:init_len, :, :, :], action[:, 0:init_len :])
			output, q_output, _ = self.ar_forward(latent[:, init_len:init_len+1, :, :, :], action[:, init_len:, :], h)
			#loss = weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay)
			q_loss = change_mse(q_output, latent[:, init_len:, :, :, :])
			with torch.no_grad():
				#q_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay)
				loss = change_mse(output, latent[:, init_len:, :, :, :])
				target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
				pred = self.compute_classification_target(q_output)
				accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
			q_loss.backward()
			optim.step()
			total_q_loss += q_loss.item()
			total_loss += loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
		}

	@torch.no_grad()
	def eval_rwm_style(self, loader: DataLoader, init_len: int = 3, err_decay:float=0.9) -> dict:
		self.eval()
		total_loss = 0.0
		total_q_loss = 0.0
		accuracy = 0.0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			_, _, h = self.forward(latent[:, 0:init_len, :, :, :],action[:, 0:init_len, :])
			output, q_output, _ = self.ar_forward(latent[:, init_len:init_len + 1, :, :, :],action[:, init_len:, :], h)
			#q_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay)
			q_loss = change_mse(q_output, latent[:, init_len:, :, :, :])
			total_q_loss += q_loss.item()
			#loss = weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay)
			loss = change_mse(output, latent[:, init_len:, :, :, :])
			total_loss += loss.item()
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
			pred = self.compute_classification_target(q_output)
			accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
		return {
			'mse': total_loss / len(loader),
			'qmse': total_q_loss / len(loader),
			'acc': accuracy*100 / len(loader),
		}
