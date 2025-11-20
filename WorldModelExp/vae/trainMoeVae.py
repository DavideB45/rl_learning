import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.moevae import MOEVAE
from helpers.general import best_device
from helpers.data import make_multi_view_dataloader

LATENT_DIM =64
REG_STRENGTH = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

DATA_PATH = 'data/pusher/multi_img/'
DEVICE = best_device()

if __name__ == "__main__":
	moe_vae = MOEVAE(latent_dim=LATENT_DIM, device=DEVICE, learn_gating=True)
	print(moe_vae)
	num_params = sum(p.numel() for p in moe_vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader, val_loader = make_multi_view_dataloader(data_dir=DATA_PATH, batch_size=64, test_split=0.2)
	
	optimizer = torch.optim.Adam(moe_vae.parameters(), lr=LEARNING_RATE)
	for epoch in range(NUM_EPOCHS):
		moe_vae.train()
		train_loss = moe_vae.train_epoch(train_loader, optimizer, REG_STRENGTH)['avg_loss']
		moe_vae.eval()
		val_loss = moe_vae.eval_epoch(val_loader, REG_STRENGTH)['avg_loss']
		print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

	model_path = "moe_vae_model.pth"
	torch.save(moe_vae.state_dict(), model_path)
	print(f"Trained MOE VAE model saved to {model_path}")
			