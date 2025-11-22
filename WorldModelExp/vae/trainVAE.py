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
from global_var import CURRENT_ENV

LATENT_DIM = 512
REG_STRENGTH = 0.5
NUM_EPOCHS = 20
LEARNING_RATE = 2e-3

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()


if __name__ == "__main__":
	if basic:
		vae = VAE(latent_dim=LATENT_DIM, device=DEVICE)
	else:
		vae = VAE(codebook_size=256,
				code_depth=8,
				latent_dim=4,
				commitment_cost=0.25,
				device=DEVICE)
	print(vae)
	num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader, val_loader = make_img_dataloader(data_dir=DATA_PATH, batch_size=64, test_split=0.2)
	reg_strength = 0.0
	for epoch in range(NUM_EPOCHS):
		reg_strength = REG_STRENGTH * min(1.0, (epoch + 1) / 10.0)
		print("-" * 25 + f" {(epoch + 1):02}/{NUM_EPOCHS} " + "-" * 25)
		vae.train()
		tr_loss = vae.train_epoch(train_loader, torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE), reg_strength)
		vae.eval()
		val_loss = vae.eval_epoch(val_loader, reg_strength)
		colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
		for i, key in enumerate(tr_loss):
			color = colors[i % len(colors)]
			reset = '\033[0m'
			print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")
	
	model_path = "vae_model.pth"
	torch.save(vae.state_dict(), model_path)
	print(f"Trained VAE model saved to {model_path}")
	