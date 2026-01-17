import torch
import torch.nn as nn
import torch.nn.functional as F

# This code was taken from a torch tutorial
class MultiHeadAttention(nn.Module):
	"""
	Computes multi-head attention. Supports nested or padded tensors.

	Args:
		E_q (int): Size of embedding dim for query
		E_k (int): Size of embedding dim for key
		E_v (int): Size of embedding dim for value
		E_total (int): Total embedding dim of combined heads post input projection. Each head
			has dim E_total // nheads
		nheads (int): Number of heads
		dropout (float, optional): Dropout probability. Default: 0.0
		bias (bool, optional): Whether to add bias to input projection. Default: True
	"""

	def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int, nheads: int, dropout: float = 0.0, bias=True, device=None, dtype=None):
		factory_kwargs = {"device": device, "dtype": dtype}
		super().__init__()
		self.nheads = nheads
		self.dropout = dropout
		self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
		if self._qkv_same_embed_dim:
			self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
		else:
			self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
			self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
			self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
		E_out = E_q
		self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
		assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
		self.E_head = E_total // nheads
		self.bias = bias

	def forward(
		self,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attn_mask=None,
		is_causal=False,
	) -> torch.Tensor:
		"""
		Forward pass; runs the following process:
			1. Apply input projection
			2. Split heads and prepare for SDPA (scaled fot product attention)
			3. Run SDPA
			4. Apply output projection

		Args:
			query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
			key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
			value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
			attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
			is_causal (bool, optional): Whether to apply causal mask. Default: False

		Returns:
			attn_output (torch.Tensor): output of shape (N, L_t, E_q)
		"""
		# Step 1. Apply input projection
		if self._qkv_same_embed_dim:
			if query is key and key is value:
				result = self.packed_proj(query)
				query, key, value = torch.chunk(result, 3, dim=-1)
			else:
				q_weight, k_weight, v_weight = torch.chunk(
					self.packed_proj.weight, 3, dim=0
				)
				if self.bias:
					q_bias, k_bias, v_bias = torch.chunk(
						self.packed_proj.bias, 3, dim=0
					)
				else:
					q_bias, k_bias, v_bias = None, None, None
				query, key, value = (
					F.linear(query, q_weight, q_bias),
					F.linear(key, k_weight, k_bias),
					F.linear(value, v_weight, v_bias),
				)

		else:
			query = self.q_proj(query)
			key = self.k_proj(key)
			value = self.v_proj(value)

		# Step 2. Split heads and prepare for SDPA
		# reshape query, key, value to separate by head
		# (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
		query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
		# (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
		key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
		# (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
		value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

		# Step 3. Run SDPA
		# (N, nheads, L_t, E_head)
		attn_output = F.scaled_dot_product_attention(
			query, key, value, dropout_p=self.dropout if self.training else 0, is_causal=is_causal
		)
		# (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
		attn_output = attn_output.transpose(1, 2).flatten(-2)

		# Step 4. Apply output projection
		# (N, L_t, E_total) -> (N, L_t, E_out)
		attn_output = self.out_proj(attn_output)

		return attn_output

# This is again code from me
class Transformer(nn.Module):
	"""
	The standard transformer layer, I guess actually a decoder only, 
	you give something in and it gave something out.
	I think the shapes should be something like (Batch, Len, Size) and the output is gonna be the same
	"""

	def __init__(self, size:int, n_heads:int, dropout=0.1, device:torch.device=None):
		super().__init__()
		self.size = size

		self.norm1 = nn.LayerNorm(size)
		self.mha = MultiHeadAttention(size, size, size, size, nheads=n_heads, dropout=dropout, device=device)
		self.norm2 = nn.LayerNorm(size) 
		self.fnn = nn.Sequential(
			nn.Linear(size, size*n_heads),
			nn.GELU(),
			nn.Linear(size*n_heads, size)
		)
		print('[WARNING] remember to compile')

	def forward(self, sequence:torch.Tensor) -> torch.Tensor:
		out = self.norm1(sequence)
		out = self.mha(out, out, out) + sequence
		out = self.norm2(out)
		return self.fnn(out) + out

class PositionalEncoding(nn.Module):
	"""
	PositionalEncoding Class from yet another pytorch tutorial:
	available @ https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
	"""

	def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 500):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, emb_size, 2) * (-torch.math.log(10000.0) / emb_size))
		pe = torch.zeros(max_len, 1, emb_size)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Arguments:
			x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
		"""
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)

class TransformerEncoder(nn.Module):
	"""
	This block can be used to preprocess an array before giving a sequence to a Transformer,
	It also apply a positional encoding to the given sequence
	"""

	def __init__(self, in_size:int, out_size:int, dropout:float = 0.1, max_len:int = 500):
		super().__init__()

		self.transform = nn.Sequential(
			nn.Linear(in_features=in_size, out_features=in_size*2),
			nn.GELU(),
			nn.LayerNorm(),
			nn.Linear(in_features=in_size*2, out_features=in_size),
			nn.GELU(),
			nn.LayerNorm(),
		)
		self.project = nn.Linear(in_features=in_size, out_features=out_size)
		self.positional_encode = PositionalEncoding(out_size, dropout, max_len)

	def forward(self, sequence:torch.Tensor) -> torch.Tensor:
		'''
		Transform the input sequence (each element independently) and add positional encoding
		
		:param sequence: The sequence to encode
		:type sequence: torch.Tensor
		:return: the encoded sequence
		:rtype: Tensor
		'''
		out = self.transform(sequence) + sequence
		out = self.project(out)
		return self.positional_encode(out)