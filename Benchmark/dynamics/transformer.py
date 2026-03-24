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
from helpers.metrics import weighted_mse

class TransformerArc(nn.Module):
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
		in_size = self.w_h*self.w_h*self.cd + act_size
		self.max_seq_len = max_seq_len

		self.encode = TransformerEncoder(
			in_size=in_size,
			out_size=emb_size,
			dropout=dropout,
			max_len=max_seq_len
		)

		self.transform = nn.Sequential(
			*[Transformer(emb_size, n_heads, dropout, device) for _ in range(n_transformer)]
		)

		self.decode_img = TransformerDecoderRD(
			in_size=emb_size,
			out_size=in_size - act_size
		)

		self.guess_reward = TransformerDecoderRD(
			in_size=emb_size,
			out_size=1
		)

		self.guess_token = nn.Parameter(torch.randn(1, 1, in_size))

		self.device = device
		self.to(device)
		self.compile()

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
		input = input.view(b, s, self.w_h, self.w_h, self.cd)
		input = input.permute(0, 1, 4, 2, 3).contiguous()
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
		:param action: The action done at each time step (B,S,W,H,D))
		:return: The prediction, the prediction quantized, the predicted reward, the last `embedding`
		'''
		sequence = self.flatten_rep(sequence.detach())
		sequence_ = torch.cat([sequence, action], dim=-1)
		guess_token = self.guess_token.expand(sequence_.size(0), -1, -1)
		sequence_ = torch.cat([sequence_, guess_token], dim=1)
		sequence_ = self.transform(self.encode(sequence_))
		last = sequence_[:, -1:, :] # hopefully (B,1,E)
		decoded_last = self.decode_img.forward(last) + sequence[:, -1:, :]
		reward = self.guess_reward(last)
		decoded_last = self.unflatten_rep(decoded_last, 1)
		decoded_last = decoded_last.squeeze()
		quantiz_last = self.vq.quantizer.quantize_fixed_space(decoded_last)
		warnings.warn('quantize fixed space')
		decoded_last = decoded_last.unsqueeze(1)
		quantiz_last = quantiz_last.unsqueeze(1)
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

		_, len, _ = action.shape
		start = min(len - self.max_seq_len - 1, 0)
		x = input[:, start:].detach()
		action = action[:, start:]

		_, len, _ = action.shape
		_, init, _, _, _ = input.shape
		#print(x.shape, action.shape)
		for t in range(init-1 , len):
			begin = max(t - self.max_seq_len + 2, 0)
			#print(begin, t, len, self.max_seq_len)
			# the full sequence should be used each time
			out, q_out, rew, _ = self.forward(x[:, begin:t+1, :], action[:, begin:t+1, :])
			preds.append(out)
			preds_q.append(q_out)
			rewards.append(rew)
			x = torch.cat([x, q_out], dim=1)

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
	
	def train_rwm_style(self, loader:DataLoader, optim:Optimizer, init_len:int=3, err_decay:float=0.95) -> dict:
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
			rewards_target = batch['reward'].to(self.device)
			optim.zero_grad()

			output, q_output, rewards = self.ar_forward(latent[:, :init_len+1, :, :, :], action)
			
			lat_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay)
			rew_loss = weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay)
			with torch.no_grad():
				q_loss = weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay)
				target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
				pred = self.compute_classification_target(q_output)
				accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
				first_accuracy += (target[:, 0:1, :].argmax(dim=-1) == pred[:, 0:1, :].argmax(dim=-1)).float().mean().item()
			loss = lat_loss + rew_loss
			loss.backward()
			optim.step()

			total_q_loss += q_loss.item()
			total_loss += lat_loss.item()
			total_reward_loss += rew_loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy*100/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}
	
	@torch.no_grad()
	def eval_rwm_style(self, loader:DataLoader, init_len:int=3, err_decay:float=0.95) -> dict:
		self.train()
		total_loss = 0
		total_q_loss = 0
		total_prop_loss = 0
		total_reward_loss = 0
		accuracy = 0.0
		first_accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device)
			action = batch['action'].to(self.device)
			rewards_target = batch['reward'].to(self.device)

			output, q_output, rewards = self.ar_forward(latent[:, :init_len+1, :, :, :], action)
			
			total_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], output, err_decay)
			total_q_loss += weighted_mse(latent[:, init_len + 1:, :, :, :], q_output, err_decay)
			total_reward_loss += weighted_mse(rewards_target[:, init_len:].unsqueeze(-1), rewards, err_decay).item()
			target = self.compute_classification_target(latent[:, init_len + 1:, :, :, :])
			pred = self.compute_classification_target(q_output)
			accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
			first_accuracy += (target[:, 0:1, :].argmax(dim=-1) == pred[:, 0:1, :].argmax(dim=-1)).float().mean().item()

		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
			'prop_mse': total_prop_loss/len(loader),
			'first_acc': first_accuracy*100/len(loader),
			'reward_mse': total_reward_loss/len(loader)
		}