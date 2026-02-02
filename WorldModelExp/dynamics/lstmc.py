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
from helpers.metrics import weighted_mse, weighted_ce, weighted_categorical_kl, pred_accuracy

class LSTMQClass(nn.Module):
	def __init__(self, quantizer:VQVAE, device:torch.device, action_dim:int, prop_dim:int, hidden_dim:int=512):
		super(LSTMQClass, self).__init__()
		self.w_h = quantizer.latent_dim
		self.d = quantizer.code_depth
		self.classes = quantizer.codebook_size
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
			nn.Linear(self.hidden_dim*3, hidden_dim),
			#nn.Linear(self.hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU()
		)
		self.lstm = LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
		self.out_emb_fc = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, self.classes*self.w_h*self.w_h),
			#nn.Sigmoid()
		)
		self.out_prop_fc = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, prop_dim),
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
			input (torc.Tensor): Input tensor shape (Batch, Seq_len, Width*Height*Classes)
			s (int): The lenght of the original sequence
		Returns:
			torch.Tensor: the flattened input (Batch, Seq_len, Depth, Width, Height)
		'''
		b = input.size(0)
		w = self.w_h
		h = w
		c = self.classes # codebook size
		d = self.d # depth

		# input = input.view(b, s, w, h, c) # Batch, Seq_len, Width, Height, Classes
		input = input.view(b*s*w*h, c) # Batch*Seq_len*Width*Height, Classes
		input = self.quantizer.quantizer.vec_from_prob(input) # Batch*Seq_len*Width*Height, Depth
		input = input.view(b, s, w, h, d) # Batch, Seq_len, Width, Height, Depth
		input = input.permute(0, 1, 4, 2, 3).contiguous() # Batch, Seq_len, Depth, Width, Height
		return input
	
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
		w = self.w_h
		h = w
		c = self.classes # codebook size
		d = self.d # depth

		target = target.contiguous().view(b*s, d, w, h) # (B*S, D, W, H)
		#target = self.quantizer.quantizer.onehot_from_vec(target) # (B*S, C, W, H)
		target = self.quantizer.quantizer.get_index_probabilities(target)
		target = target.view(b, s, c, w, h).contiguous() # (B, S, C, W, H)
		target = target.permute(0, 1, 3, 4, 2) # (B, S, W, H, C)
		return target

	def forward(self, input:torch.Tensor, action:torch.Tensor, prop:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture
		
		:param input: Input tensor shape (Batch, Seq_len, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action
		:type action: torch.Tensor
		:param prop: a tensor representing the robot proprioception
		:type prop: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence as categorical distribution (Batch, Seq_len, Classes*Width*Height) | the hidden state of the LSTM
		:rtype: tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]
		'''
		input = self.flatten_rep(input.detach())
		new_rep = self.rep_fc(input.detach())
		prop = self.pro_fc(prop)
		action = self.act_fc(action)
		output = torch.cat([new_rep, action, prop], dim=-1)
		skip_output = self.merge_fc(output) #(B, Seq_len, Hidden_dim)

		if h is None:
			h = (torch.zeros(1, input.size(0), self.hidden_dim).to(input.device),
				 torch.zeros(1, input.size(0), self.hidden_dim).to(input.device))
		output, h = self.lstm(skip_output, h)

		output = output + skip_output #(B, Seq_len, Hidden_dim)
		latent = self.out_emb_fc(output) #(B, Seq_len, Width*Height*Classes)
		latent_q = self.unflatten_rep(latent, input.size(1)) # Batch, Seq_len, Depth, Width, Height
		prop_out = self.out_prop_fc(output) #(B, Seq_len, Prop_dim)
		
		return latent, latent_q, prop_out, h
	
	def ar_forward(self, input:torch.Tensor, action:torch.Tensor, prop:torch.Tensor, h=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		'''
		Do the forward pass in an LSTM architecture in an autoregressive fashion
		
		:param input: Input tensor shape (Batch, 1, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action (Batch, Seq_len, Act_size)
		:type action: torch.Tensor
		:param prop: a tensor representing the robot proprioception (Batch, 1, Prop_size)
		:type prop: torch.Tensor
		:param h: initial hidden state (optional)
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) | the hidden state of the LSTM
		:rtype: tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]
		'''
		batch, len, _ = action.shape
		preds_q = []
		preds = []
		preds_prop = []

		if h is None:
			h = (torch.zeros(1, batch, self.hidden_dim).to(self.device),
				 torch.zeros(1, batch, self.hidden_dim).to(self.device))
		x = input.detach()
		for t in range(len):
			out, q_out, p_out, h = self.forward(x, action[:, t:t+1, :], prop, h)
			preds_prop.append(p_out)
			preds_q.append(q_out)
			preds.append(out)
			x = q_out
			prop = p_out
			
		preds_q = torch.cat(preds_q, dim=1)
		preds = torch.cat(preds, dim=1)
		preds_prop = torch.cat(preds_prop, dim=1)
		return preds, preds_q, preds_prop, h
	
	def compute_ce(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
		'''
		Compute the mean square error for each time step and weight it by the decay factor
		
		:param pred: the generated sequence (Batch, Seq_len, Width*Height*Classes)
		:type pred: torch.Tensor
		:param target: the original sequence (Batch, Seq_len, Width*Height*Classes)
		:type target: torch.Tensor
		:return: the computed error
		:rtype: Tensor
		'''
		b = target.size(0)
		s = target.size(1)
		w = self.w_h
		h = w
		c = self.classes # codebook size
		
		pred = pred.view(b*s*w*h, c)
		#target = target.view(b*s*w*h, c)
		target_indices = torch.argmax(target, dim=-1).view(b * s * w * h)
		return F.cross_entropy(pred, target_indices, reduction='mean')
	
	def train_rwm_style(self, loader:DataLoader, optim:Optimizer, init_len:int=3, err_decay:float=0.9, useKL:bool=False) -> dict:
		self.train()
		total_ce = 0
		total_q_loss = 0
		total_prop_loss = 0
		accuracy = 0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			proprioception = batch['proprioception'].to(self.device)
			optim.zero_grad()
			
			_, _, _, h = self.forward(latent[:, 0:init_len, :, :, :], action[:, 0:init_len :], proprioception[:, 0:init_len, :])
			output, q_output, prop_out, _ = self.ar_forward(latent[:, init_len:init_len+1, :, :, :], action[:, init_len:, :], proprioception[:, init_len:init_len+1, :], h)
			
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :]).detach()
			if useKL:
				class_loss = weighted_categorical_kl(output, target, self.w_h, self.classes, err_decay)
			else:
				class_loss = weighted_ce(output, target, self.w_h, self.classes, err_decay)
			total_prop_loss += weighted_mse(proprioception[:, init_len + 1:, :], prop_out, err_decay)

			with torch.no_grad():
				total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay).item()
				accuracy += pred_accuracy(output, target, self.w_h, self.classes).item()
				first_accuracy += pred_accuracy(output[:, 0:1, :], target[:, 0:1, :], self.w_h, self.classes).item()
			loss = class_loss + total_prop_loss
			loss.backward()
			optim.step()
			total_ce += class_loss.item()
			total_prop_loss += total_prop_loss.item()
		return {
			'ce': total_ce/len(loader),
			'acc': accuracy/len(loader),
			'mse': total_q_loss/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy/len(loader)
		}

	@torch.no_grad()
	def eval_rwm_style(self, loader: DataLoader, init_len: int = 3, err_decay:float=0.9, useKL:bool=False) -> dict:
		self.eval()
		total_ce = 0
		total_q_loss = 0
		total_prop_loss = 0
		accuracy = 0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			proprioception = batch['proprioception'].to(self.device)

			_, _, _, h = self.forward(latent[:, 0:init_len, :, :, :],action[:, 0:init_len, :], proprioception[:, 0:init_len, :])
			output, q_output, prop_output, _ = self.ar_forward(latent[:, init_len:init_len + 1, :, :, :],action[:, init_len:, :], proprioception[:, init_len:init_len + 1, :], h)

			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
			total_prop_loss += weighted_mse(proprioception[:, init_len + 1:, :], prop_output, err_decay).item()
			if useKL:
				total_ce += weighted_categorical_kl(output, target, self.w_h, self.classes, err_decay).item()
			else:
				total_ce += weighted_ce(output, target, self.w_h, self.classes, err_decay).item()
			total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay).item()
			accuracy += pred_accuracy(output, target, self.w_h, self.classes).item()
			first_accuracy += pred_accuracy(output[:, 0:1, :], target[:, 0:1, :], self.w_h, self.classes).item()
			
		return {
			'ce': total_ce/len(loader),
			'acc': accuracy/len(loader),
			'mse': total_q_loss/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy/len(loader)
		}
