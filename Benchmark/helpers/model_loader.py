import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from dynamics.lstmc import LSTMQClass

import torch


def load_vq_vae(env:dict, codebook_size:int, code_depth:int, latent_dim:int, ema_mode:bool, smooth:bool, device) -> VQVAE:
	model = VQVAE(
		codebook_size=codebook_size,
		code_depth=code_depth,
		latent_dim=latent_dim,
		commitment_cost=0.25,
		ema_mode=ema_mode,
		device=device
	)
	model_path = env['models'] + f"vq_{latent_dim}_{code_depth}_{codebook_size}_{ema_mode}_smooth{smooth}.pth"
	print(f'Loading {f"vq_{latent_dim}_{code_depth}_{codebook_size}_{ema_mode}_smooth{smooth}.pth"}')
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model


def load_lstm_quantized(env:dict, vq:VQVAE, device:torch.device, hidden_dim:int, tf:bool=False, cl:bool=False, kl:bool=False) -> LSTMQuantized | LSTMQClass:
	if cl:
		model = LSTMQClass(vq, device, env['a_size'], 4, hidden_dim)
	else:
		model = LSTMQuantized(vq, device, env['a_size'], 4, hidden_dim)
	d = vq.code_depth
	w_h = vq.latent_dim
	s = vq.codebook_size
	if cl:
		model_path = env['models'] + f"lstmqc_{model.hidden_dim}_{w_h}_{d}_{s}"
	else:
		model_path = env['models'] + f"lstmq_{model.hidden_dim}_{w_h}_{d}_{s}"
	if tf:
		model_path += '_tf'
	if kl:
		model_path += '_kl'
	model_path += ".pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model



def save_vq_vae(env:dict, model:VQVAE, smooth:bool) -> str:
	model_path = env['models'] + f"vq_{model.latent_dim}_{model.code_depth}_{model.codebook_size}_{model.ema_mode}_smooth{smooth}.pth"
	torch.save(model.state_dict(), model_path)
	return model_path

def save_lstm_quantized(env:dict, model:LSTMQuantized | LSTMQClass, tf:bool=False, cl:bool=False, kl:bool=False) -> str:
	d = model.d
	w_h = model.w_h
	s = model.quantizer.codebook_size
	if cl:
		model_path = env['models'] + f"lstmqc_{model.hidden_dim}_{w_h}_{d}_{s}"
	else:
		model_path = env['models'] + f"lstmq_{model.hidden_dim}_{w_h}_{d}_{s}"
	if tf:
		model_path += '_tf' # teacher forcing
	if kl:
		model_path += '_kl' # kl usage as target
	model_path += ".pth"
	torch.save(model.state_dict(), model_path)
	return model_path
