import torch
import torch.nn as nn
import torch.nn.functional as F
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
			out_size=in_size
		)

		self.guess_token = nn.Parameter(torch.randn(1, 1, in_size))

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
		sequence = torch.cat([sequence, action], dim=-1)
		guess_token = self.guess_token.expand(sequence.size(0), -1, -1)
		sequence = torch.cat([sequence, guess_token], dim=1)
		sequence = self.transform(self.encode(sequence))
		last = sequence[:, -2:-1, :] # hopefully (B,1,E)
		decoded_last = self.decode(last)
		quantiz_last = self.vq.quantizer.quantize_fixed_space(decoded_last)
		warnings.warn('quantize fixed space')
		return decoded_last, quantiz_last, last
