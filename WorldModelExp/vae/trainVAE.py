import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

basic = False
if basic:
	from vae.myVae import CVAE as VAE
else:
	from vae.vqVae import VQVAE as VAE
from helpers.general import best_device
from helpers.data import make_img_dataloader
from helpers.model_loader import save_base_vae, save_vq_vae
from global_var import CURRENT_ENV

LATENT_DIM = 32
REG_STRENGTH = 0.5

NUM_EPOCHS = 50
LEARNING_RATE = 5e-4

LATENT_DIM_VQ = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 64
EMA_MODE = True

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()


if __name__ == "__main__":
	if basic:
		vae = VAE(latent_dim=LATENT_DIM, device=DEVICE)
	else:
		vae = VAE(codebook_size=CODEBOOK_SIZE,
				code_depth=CODE_DEPTH,
				latent_dim=LATENT_DIM_VQ,
				commitment_cost=0.25,
				device=DEVICE,
				ema_mode=EMA_MODE
				)
	#print(vae)
	num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader, val_loader = make_img_dataloader(data_dir=DATA_PATH, batch_size=64, test_split=0.2)
	print(f"Training on {len(train_loader.dataset)} images, validating on {len(val_loader.dataset)} images.")
	reg_strength = 0.0
	best_val_loss = float('inf')
	for epoch in range(NUM_EPOCHS):
		reg_strength = REG_STRENGTH * min(1.0, (epoch + 1) / 40.0)
		print("-" * 25 + f" {(epoch + 1):02}/{NUM_EPOCHS} " + "-" * 25)
		vae.train()
		tr_loss = vae.train_epoch(train_loader, torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE), reg_strength)
		vae.eval()
		val_loss = vae.eval_epoch(val_loader, REG_STRENGTH)
		colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
		reset = '\033[0m'
		if val_loss['total_loss'] < best_val_loss:
			best_val_loss = val_loss['total_loss']
			# save the best model
			if basic:
				save_base_vae(CURRENT_ENV, vae, REG_STRENGTH)
			else:
				save_vq_vae(CURRENT_ENV, vae)
			print(f"{colors[-1]}  New best model saved!{reset}")
		for i, key in enumerate(tr_loss):
			color = colors[i % len(colors)]
			print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")
	