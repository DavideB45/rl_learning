import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import warnings

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from dynamics.blocks_tr import Transformer, TransformerEncoder, TransformerDecoderRD
from helpers.metrics import weighted_mse, weighted_categorical_kl, weighted_ce, pred_accuracy

class TransformerArcC(nn.Module):
	'''
	An (almost) fully customized implementation fo a transformer architecture 
	
	The main components of this architecture are:
		- an encoder that transform the input and apply the positional encoding (from the attention is all you need paper)
		- a transformer (decoder only) backbone that follows the GPT2 architecture
		- a decoder that brings the transformer predicted values in the desired observation space
	
	By default the architecture is Batch Frist to simplify integration with other architecture and code reusability, for the moment
	there is no plan of allowing a non Batch First mode.
	'''

	def __init__(self, act_size:int, vq:VQVAE, emb_size:int, max_seq_len:int, n_heads:int, n_transformer:int, dropout:float, device:torch.device):
		'''
		Create a Transformer based model
		
		:param act_size: The size of the action space
		:param vq: The VQ-VAE used to generate the data, used to reshape the data in the correct way and quantize prediction
		:param emb_size: The size of the embedding that will be used inside the transformer
		:param max_seq_len: The max sequence length (used to initialize the positional encoding)
		:param n_heads: number of heads to use inside the transformers module
		:param n_transformer: number of replications of the transformer module \
		(Higher may yeald a better accuracy but will also linearly increase the inference time)
		:param dropout: Dropout parameter for dropout regularization after the encoder and in the attention layers
		'''
		super().__init__()

		self.vq = vq
		self.w_h = vq.latent_dim
		self.cd = vq.code_depth
		self.cs = vq.codebook_size
		in_size = self.w_h*self.w_h*self.cd
		self.max_seq_len = max_seq_len
		self.emb_size = emb_size

		self.rep_fc = nn.Sequential(
			nn.Linear(in_size, emb_size),
			nn.LeakyReLU()
		)
		self.act_fc = nn.Sequential(
			nn.Linear(act_size, emb_size),
			nn.LeakyReLU(),
			nn.LayerNorm(emb_size),
			nn.Linear(emb_size, emb_size),
			nn.LeakyReLU(),
			nn.LayerNorm(emb_size)
		)
		self.encode = TransformerEncoder(
			in_size=emb_size,
			out_size=emb_size,
			dropout=dropout,
			max_len=max_seq_len
		)

		self.transform = nn.Sequential(
			*[Transformer(emb_size, n_heads, dropout, device) for _ in range(n_transformer)]
		)

		self.decode_img = TransformerDecoderRD(
			in_size=emb_size,
			out_size=self.cs*self.w_h*self.w_h
		)

		self.guess_reward = TransformerDecoderRD(
			in_size=emb_size,
			out_size=1
		)

		self.guess_token = nn.Parameter(torch.randn(1, 1, in_size))

		self.device = device
		self.to(device)
		self.compile()
	
	def param_count(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def flatten_rep(self, input:torch.Tensor) -> torch.Tensor:
		'''
		Flatten the input from (B,S,D,W,H) -> (B,S,W*H*D)
		'''
		input = input.permute(0, 1, 3, 4, 2).contiguous() # (B, S, W, H, D)
		input = input.view(input.size(0), input.size(1), -1) # (B, S, W*H*D)
		return input
	
	def unflatten_rep(self, input:torch.Tensor, s:int) -> torch.Tensor:
		'''
		Takes as input a flat tensor generated from the model and brings it to the correct shape \
		s stands for sequence length
		'''
		b = input.size(0)
		w = self.w_h
		h = w
		c = self.cs # codebook size
		d = self.cd # depth

		# input = input.view(b, s, w, h, c) # Batch, Seq_len, Width, Height, Classes
		input = input.view(b*s*w*h, c) # Batch*Seq_len*Width*Height, Classes
		input = self.vq.quantizer.vec_from_prob(input) # Batch*Seq_len*Width*Height, Depth
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

		target = target.contiguous().view(b*s, self.cd, self.w_h, self.w_h) # (B*S, D, W, H)
		target = self.vq.quantizer.get_index_probabilities(target)
		target = target.view(b, s, self.cs, self.w_h, self.w_h).contiguous() # (B, S, C, W, H)
		target = target.permute(0, 1, 3, 4, 2) # (B, S, W, H, C)
		return target
	
	def forward(self, sequence:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		'''
		Do a forward pass generating a single token
		Since this architecture supports decoder only model, 
		only the generated token and it's decoding will be returned
		
		:param sequence: The sequence of perceptions (B,S,W,H,D)
		:param action: The action done at each time step (B,S,N))
		:return: The prediction, the prediction quantized, the predicted reward, the last `embedding`
		'''
		if sequence.shape[1] != action.shape[1]:
			raise IndexError(f'sequence len != action len {sequence.shape[1]} != {action.shape[1]}')
		sequence_ = self.flatten_rep(sequence.detach())
		sequence_ = self.rep_fc(sequence_)
		#sequence_ = torch.cat([sequence, action], dim=-1)
		#guess_token = self.guess_token.expand(sequence_.size(0), -1, -1)
		guess_token = self.act_fc(action[:, -1, :])
		sequence_ = torch.cat([sequence_, guess_token.unsqueeze(1)], dim=1)
		skip = self.encode(sequence_)
		sequence_ = self.transform(skip)
		last = sequence_[:, -1:, :] + skip[:, -2:-1, :] # hopefully (B,1,E)
		decoded_last = self.decode_img.forward(last)
		reward = self.guess_reward(last)
		quantiz_last = self.unflatten_rep(decoded_last, 1)
		decoded_last = decoded_last.unsqueeze(1)
		return decoded_last, quantiz_last, reward, last
	
	def ar_forward(self, input:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		'''
		Do the forward pass taking as input a representation and a number of action that can be of different length
		
		:param input: Input rensor of shape (Batch, init_len, Depth, Width, Height)
		:type input: torch.Tensor
		:param action: a tensor representing the robot action
		:type action: torch.Tensor
		:return: the predicted sequence before quantization | the predicted sequence quantized (Batch, Seq_len, Depth, Width, Height) | the reward
		:rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
		'''
		preds_q = []
		preds = []
		rewards = []

		msl = self.max_seq_len
		extra_len = input.shape[1] - msl + 1
		if  extra_len > 0:
			#print(f'throwing away {extra_len} initial steps to obtain len of {msl -1}')
			input = input[:, extra_len:, :]
			action = action[:, extra_len:, :]


		_, len, _ = action.shape
		for t in range(0, len-msl+2): # +1 i standard to get everything and +1 bocause later we subtract 1
			first = t 
			last = first+msl-1

			#print(first, last, len, msl, input.shape[1])
			# the full sequence should be used each time
			out, q_out, rew, _ = self.forward(input[:, first:last, :], action[:, first:last, :])
			preds.append(out)
			preds_q.append(q_out)
			rewards.append(rew)
			input = torch.cat([input, q_out], dim=1)
		preds = torch.cat(preds, dim=1)
		preds_q = torch.cat(preds_q, dim=1)
		rewards = torch.cat(rewards, dim=1)
		return preds, preds_q, rewards
	
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
		w = self.vq.latent_dim
		h = w
		c = self.vq.codebook_size
		d = self.vq.code_depth # depth

		target = target.contiguous().view(b*s, d, w, h) # (B*S, D, W, H)
		target = self.vq.quantizer.get_index_probabilities(target)
		target = target.view(b, s, c, w, h).contiguous() # (B, S, C, W, H)
		target = target.permute(0, 1, 3, 4, 2) # (B, S, W, H, C)
		return target
	
	def train_rwm_style(self, loader:DataLoader, optim:Optimizer, init_len:int=3, err_decay:float=0.9, useKL:bool=False) -> dict:
		self.train()
		total_ce = 0
		total_q_loss = 0
		total_prop_loss = 0
		total_reward_loss = 0
		accuracy = 0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device).detach()
			action = batch['action'].to(self.device).detach()
			rewards_target = batch['reward'].to(self.device)
			optim.zero_grad()

			output, q_output, rewards = self.ar_forward(latent[:, :init_len+1, :, :, :], action)
			
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :]).detach()
			if useKL:
				class_loss = weighted_categorical_kl(output, target, self.w_h, self.cs, err_decay)
			else:
				class_loss = weighted_ce(output, target, self.w_h, self.cs, err_decay)
			rew_loss = weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay)
			with torch.no_grad():
				total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay).item()
				accuracy += pred_accuracy(output, target, self.w_h, self.cs).item()
				first_accuracy += pred_accuracy(output[:, 0:1, :], target[:, 0:1, :], self.w_h, self.cs).item()
			loss = class_loss + rew_loss
			loss.backward()
			optim.step()

			total_ce += class_loss.item()
			total_reward_loss += rew_loss.item()
		return {
			'ce': total_ce/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy*100/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}
	
	@torch.no_grad()
	def eval_rwm_style(self, loader:DataLoader, init_len:int=3, err_decay:float=0.95, useKL:bool=False) -> dict:
		self.train()
		total_ce = 0
		total_q_loss = 0
		total_prop_loss = 0
		total_reward_loss = 0
		accuracy = 0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device).detach()
			action = batch['action'].to(self.device).detach()
			rewards_target = batch['reward'].to(self.device)

			output, q_output, rewards = self.ar_forward(latent[:, :init_len+1, :, :, :], action)
			
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :]).detach()
			if useKL:
				total_ce = weighted_categorical_kl(output, target, self.w_h, self.cs, err_decay).item()
			else:
				total_ce = weighted_ce(output, target, self.w_h, self.cs, err_decay).item()
			rew_loss = weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay).item()
			total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay).item()
			accuracy += pred_accuracy(output, target, self.w_h, self.cs).item()
			first_accuracy += pred_accuracy(output[:, 0:1, :], target[:, 0:1, :], self.w_h, self.cs).item()
			total_reward_loss += rew_loss.item()

		return {
			'ce': total_ce/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy*100/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}