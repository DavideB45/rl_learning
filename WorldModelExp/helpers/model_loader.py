import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from vae.myVae import CVAE
from vae.moevae import MOEVAE

import torch

def load_base_vae(env:dict, latent_dim:int, kl_b:float, device) -> CVAE:
	model = CVAE(latent_dim, device)
	if kl_b.is_integer():
		kl_b = int(kl_b)
	kl_str = str(kl_b).replace('.', '')
	model_path = env['models'] + f"vae_{latent_dim}_{kl_str}.pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def load_vq_vae(env:dict, codebook_size:int, code_depth:int, latent_dim:int, ema_mode:bool, device) -> VQVAE:
	model = VQVAE(
		codebook_size=codebook_size,
		code_depth=code_depth,
		latent_dim=latent_dim,
		commitment_cost=0.25,
		ema_mode=ema_mode,
		device=device
	)
	model_path = env['models'] + f"vq_{latent_dim}_{code_depth}_{codebook_size}_{ema_mode}.pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def load_moe_vae(env:dict, latent_dim:int, kl_b:float, concordance_reg:float, device) -> MOEVAE:
	model = MOEVAE(latent_dim, device)
	if kl_b.is_integer():
		kl_b = int(kl_b)
	kl_str = str(kl_b).replace('.', '')
	if concordance_reg.is_integer():
		concordance_reg = int(concordance_reg)
	cr_str = str(concordance_reg).replace('.', '')
	model_path = env['models'] + f"moe_{latent_dim}_{kl_str}_{cr_str}.pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def save_base_vae(env:dict, model:CVAE, kl_b:float) -> str:
	if kl_b.is_integer():
		kl_b = int(kl_b)
	kl_str = str(kl_b).replace('.', '')
	model_path = env['models'] + f"vae_{model.latent_dim}_{kl_str}.pth"
	torch.save(model.state_dict(), model_path)
	return model_path

def save_vq_vae(env:dict, model:VQVAE) -> str:
	model_path = env['models'] + f"vq_{model.latent_dim}_{model.code_depth}_{model.codebook_size}_{model.ema_mode}.pth"
	torch.save(model.state_dict(), model_path)
	return model_path