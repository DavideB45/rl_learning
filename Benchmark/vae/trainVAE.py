import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))


from vae.vqVae import VQVAE as VAE
from helpers.general import best_device
from helpers.data import make_image_dataloader_safe, get_data_path
from helpers.model_loader import save_vq_vae
from global_var import CURRENT_ENV, LATENT_DIM, CODE_DEPTH, CODEBOOK_SIZE, SMOOTH, VQ_EPOCS, VQ_LR, VQ_WD



TR_DATA = get_data_path(CURRENT_ENV['img_dir'], True, 0)
VL_DATA = get_data_path(CURRENT_ENV['img_dir'], False, 0)
DEVICE = best_device()


if __name__ == "__main__":
	vae = VAE(codebook_size=CODEBOOK_SIZE,
			code_depth=CODE_DEPTH,
			latent_dim=LATENT_DIM,
			commitment_cost=0.25,
			device=DEVICE,
			ema_mode=True
		)
	print(f"VQ-VAE with codebook size {CODEBOOK_SIZE}, code depth {CODE_DEPTH}, latent dim {LATENT_DIM}, commitment cost 0.25, EMA mode {True}")
	#print(vae)
	num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader = make_image_dataloader_safe(TR_DATA)
	val_loader = make_image_dataloader_safe(VL_DATA)
	print(f"Training on {len(train_loader.dataset)} images, validating on {len(val_loader.dataset)} images.")
	reg_strength = SMOOTH
	best_val_loss = float('inf')
	optim = torch.optim.Adam(vae.parameters(), lr=VQ_LR, weight_decay=VQ_WD)
	for epoch in range(VQ_EPOCS):
		#reg_strength = REG_STRENGTH * min(1.0, (epoch + 1) / 40.0)
		print("-" * 25 + f" {(epoch + 1):02}/{VQ_EPOCS} " + "-" * 25)
		vae.train()
		tr_loss = vae.train_epoch(train_loader, optim, reg_strength)
		vae.eval()
		val_loss = vae.eval_epoch(val_loader, reg_strength)
		colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
		reset = '\033[0m'
		if val_loss['total_loss'] < best_val_loss:
			best_val_loss = val_loss['total_loss']
			# save the best model
			save_vq_vae(CURRENT_ENV, vae, smooth=True if SMOOTH > 0 else False)
			print(f"{colors[-1]}  New best model saved!{reset}")
		for i, key in enumerate(tr_loss):
			color = colors[i % len(colors)]
			print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")
	
