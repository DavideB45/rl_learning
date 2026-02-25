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
	def __init__(self, quantizer:VQVAE, device:torch.device, action_dim:int, prop_dim:int, hidden_dim:int=512):
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
		self.pro_fc = nn.Sequential(
			nn.Linear(prop_dim, self.latent_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(self.latent_dim),
			nn.Linear(self.latent_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim)
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
		self.out_prop_fc = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, prop_dim),
		)
		self.out_reward = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, 1),
		)

		self.device = device
		self.to(device)

	def param_count(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

	def forward(self, input:torch.Tensor, action:torch.Tensor, prop:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture
		
		:param input: Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action
		:type action: torch.Tensor
		:param prop: a tensor that is the proprioception of the robot (regression)
		:type prop: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) |  the proprioception prediciton | the reward | the hidden state of the LSTM
		:rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
		'''
		input = self.flatten_rep(input)
		new_rep = self.rep_fc(input)
		action = self.act_fc(action)
		prop = self.pro_fc(prop)
		output = torch.cat([new_rep, action], dim=-1)
		skip_output = self.merge_fc(output) #(B, Seq_len, Hidden_dim)

		if h is None:
			h = (torch.zeros(1, input.size(0), self.hidden_dim).to(input.device),
				 torch.zeros(1, input.size(0), self.hidden_dim).to(input.device))
		output, h = self.lstm(skip_output, h)

		output = output + skip_output #(B, Seq_len, Hidden_dim)
		latent = self.out_fc(output) + input #(B, Seq_len, Height*Width*Depth)
		latent = self.unflatten_rep(output, input.size(1)) # (B, Seq_len, Depth, Height, Width)
		prop_out = self.out_prop_fc(output.detach()) #(B, Seq_len, Prop_dim)
		reward = self.out_reward(output)
		
		_, latent_q, _ = self.quantizer.quantize(output.reshape(-1, self.d, self.w_h, self.w_h))
		latent_q = latent_q.view(input.size(0), input.size(1), self.d, self.w_h, self.w_h)
		
		return latent, latent_q, prop_out, reward, h
	
	def ar_forward(self, input:torch.Tensor, action:torch.Tensor, prop:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture in an autoregressive fashion
		
		:param input: Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action
		:type action: torch.Tensor
		:param prop: a tensor that is the proprioception of the robot (regression)
		:type prop: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) |  the proprioception prediciton | the reward | the hidden state of the LSTM
		:rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
		'''
		batch, len, _ = action.shape
		preds_q = []
		preds = []
		preds_prop = []
		rewards = []

		if h is None:
			h = (torch.zeros(1, batch, self.hidden_dim).to(self.device),
				 torch.zeros(1, batch, self.hidden_dim).to(self.device))
		x = input.detach()
		for t in range(len):
			out, q_out, p_out, r_out, h = self.forward(x, action[:, t:t+1, :], prop, h)
			preds_prop.append(p_out)
			preds_q.append(q_out)
			preds.append(out)
			rewards.append(r_out)
			x = q_out
			prop = p_out
			
		preds_q = torch.cat(preds_q, dim=1)
		preds = torch.cat(preds, dim=1)
		preds_prop = torch.cat(preds_prop, dim=1)
		rewards = torch.cat(rewards, dim=1)
		return preds, preds_q, preds_prop, rewards, h
	
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
	

	def train_rwm_style(self, loader:DataLoader, optim:Optimizer, init_len:int=3, err_decay:float=0.9) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		total_prop_loss = 0
		total_reward_loss = 0
		accuracy = 0.0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device).detach()
			action = batch['action'].to(self.device).detach()
			proprioception = batch['proprioception'].to(self.device)
			rewards_target = batch['reward'].to(self.device)
			optim.zero_grad()

			_, _, _, _, h = self.forward(latent[:, 0:init_len, :, :, :], action[:, 0:init_len :], proprioception[:, 0:init_len, :])
			output, q_output, prop_out, rewards, _ = self.ar_forward(latent[:, init_len:init_len+1, :, :, :], action[:, init_len:, :], proprioception[:, init_len+1], h)
			
			lat_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay)
			prop_loss = weighted_mse(proprioception[:, init_len + 1:, :], prop_out, err_decay)
			rew_loss = weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay)
			with torch.no_grad():
				q_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay)
				target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
				pred = self.compute_classification_target(q_output)
				accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
				first_accuracy += (target[:, 0:1, :].argmax(dim=-1) == pred[:, 0:1, :].argmax(dim=-1)).float().mean().item()
			loss = lat_loss + prop_loss + rew_loss
			loss.backward()
			optim.step()

			total_q_loss += q_loss.item()
			total_loss += lat_loss.item()
			total_prop_loss += prop_loss.item()
			total_reward_loss += rew_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}

	@torch.no_grad()
	def eval_rwm_style(self, loader: DataLoader, init_len: int = 3, err_decay:float=0.9) -> dict:
		self.eval()
		total_loss = 0.0
		total_q_loss = 0.0
		total_prop_loss = 0
		total_reward_loss = 0
		accuracy = 0.0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			proprioception = batch['proprioception'].to(self.device)
			rewards_target = batch['reward'].to(self.device)
			
			_, _, _, _, h = self.forward(latent[:, 0:init_len, :, :, :], action[:, 0:init_len, :], proprioception[:, 0:init_len, :])
			output, q_output, prop_output, rewards, _ = self.ar_forward(latent[:, init_len:init_len + 1, :, :, :], action[:, init_len:, :], proprioception[:, init_len+1], h)
			total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay).item()
			total_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay).item()
			total_prop_loss += weighted_mse(proprioception[:, init_len + 1:, :], prop_output, err_decay).item()
			total_reward_loss += weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay).item()
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
			pred = self.compute_classification_target(q_output)
			accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
			first_accuracy += (target[:, 0:1, :].argmax(dim=-1) == pred[:, 0:1, :].argmax(dim=-1)).float().mean().item()
		return {
			'mse': total_loss / len(loader),
			'qmse': total_q_loss / len(loader),
			'acc': accuracy*100 / len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}
