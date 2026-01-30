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

		self.encode = TransformerEncoder(
			in_size=in_size,
			out_size=emb_size,
			dropout=dropout,
			max_len=max_seq_len
		)

		self.transform = nn.Sequential(
			*[Transformer(emb_size, n_heads, dropout, device) for _ in range(n_transformer)]
		)

		self.decode = TransformerDecoderRD(
			in_size=emb_size,
			out_size=in_size - act_size
		)

		self.guess_token = nn.Parameter(torch.randn(1, 1, in_size))

		self.device = device
		self.to(device)

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
	
	def forward(self, sequence:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		'''
		Do a forward pass generating a single token
		Since this architecture supports decoder only model, 
		only the generated token and it's decoding will be returned
		
		:param sequence: The sequence of perceptions (B,S,W,H,D)
		:param action: The action done at each time step (B,S,W,H,D))
		:return: The prediction, the prediction quantized, the last `embedding`
		'''
		sequence = self.flatten_rep(sequence.detach())
		sequence_ = torch.cat([sequence, action], dim=-1)
		guess_token = self.guess_token.expand(sequence_.size(0), -1, -1)
		sequence_ = torch.cat([sequence_, guess_token], dim=1)
		sequence_ = self.transform(self.encode(sequence_))
		last = sequence_[:, -1:, :] # hopefully (B,1,E)
		decoded_last = self.decode.forward(last) + sequence[:, -1:, :]
		decoded_last = self.unflatten_rep(decoded_last, 1)
		decoded_last = decoded_last.squeeze()
		quantiz_last = self.vq.quantizer.quantize_fixed_space(decoded_last)
		warnings.warn('quantize fixed space')
		decoded_last = decoded_last.unsqueeze(1)
		quantiz_last = quantiz_last.unsqueeze(1)
		return decoded_last, quantiz_last, last

	def train_epoch(self, loader:DataLoader, optim:Optimizer) -> dict:
		'''
		There is not much to say about this funciton, it is used to train the model 
		in an autoregressive way, using the special loss funciton
		
		:param loader: A dataloader for the data
		:param optim: The optimizer to use in training
		:return: Measurements about the errors in a dictionary (mse, qmse and acc)
		'''
		self.train()
		total_loss = 0
		total_q_loss = 0
		accuracy = 0
		for batch in loader:
			latent = batch['latent'].to(self.device).detach()
			action = batch['action'].to(self.device).detach()
			optim.zero_grad()
			output, q_output, _ = self.forward(latent[:, :-1, :, :, :], action)
			loss = change_mse(output, latent[:, -1:, :, :, :], latent[:, -2:-1, :, :, :])
			with torch.no_grad():
				total_q_loss += F.mse_loss(latent[:, -1:, :, :, :], output, reduction='mean')
				target = self.compute_classification_target(latent[:, -1:, :, :, :])
				pred = self.compute_classification_target(q_output)
				accuracy += (target.argmax(dim=-1) == pred.argmax(dim=-1)).float().mean().item()
			loss.backward()
			optim.step()
			total_loss += loss.item()
		return {
			'mse': total_loss/len(loader),
			'qmse': total_q_loss/len(loader),
			'acc': accuracy*100/len(loader),
		}

def change_mse(pred:torch.Tensor, target:torch.Tensor, prev_target:torch.Tensor) -> torch.Tensor:
	'''
	Special loss used to compute MSE error that is weighted based on the change of the 
	value from one time stamp  to the next, more changes means the error will be valued more

	:param pred: the generated element (Batch, 1, Width,Height,Classes)
	:param target: the original element (Batch, 1, Width,Height,Classes)
	:param prev_target: the element before the predicted one (Batch, 1, Width,HeightClasses)
	:return: the computed error based on input change
	'''
	err_weight = target - prev_target
	err_weight = torch.abs(err_weight)
	max_e = torch.max(err_weight)
	max_e = max(max_e, 1)
	err_weight = ((err_weight/max_e)*9 + 1).detach()
	#loss = (((pred - target) ** 2)*err_weight).mean()
	loss = (((pred - target) ** 2)).mean()
	return loss