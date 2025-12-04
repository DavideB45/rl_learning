import torch
import json

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.moevae import MOEVAE
from helpers.general import best_device
from helpers.data import make_multi_view_dataloader

LATENT_DIM = 64
REG_STRENGTH = 1.0
CONCORD_STRENGTH = 0.0
NUM_EPOCHS = 60
LEARNING_RATE = 5e-4

DATA_PATH = 'data/pusher/multi_img/'
DEVICE = best_device()

if __name__ == "__main__":
	moe_vae = MOEVAE(
		latent_dim=LATENT_DIM,
		device=DEVICE,
		learn_gating=False,
	)
	#print(moe_vae)
	num_params = sum(p.numel() for p in moe_vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader, val_loader = make_multi_view_dataloader(
		data_dir=DATA_PATH,
		batch_size=64,
		test_split=0.2
	)
	
	optimizer = torch.optim.Adam(moe_vae.parameters(), lr=LEARNING_RATE)
	losses_history = {
		"train_loss": [],
		"val_loss": []
	}
	reg_str = 0.0
	concord_str = 0.0
	for epoch in range(NUM_EPOCHS):
		reg_str = REG_STRENGTH * min(1.0, (epoch) / 20.0)
		concord_str = CONCORD_STRENGTH * min(1.0, (epoch) / 20.0)
		print("-" * 25 + f" {(epoch + 1):02}/{NUM_EPOCHS} " + "-" * 25)
		moe_vae.train()
		tr_loss = moe_vae.train_epoch(train_loader, optimizer, reg_str, concord=concord_str)
		moe_vae.eval()
		val_loss = moe_vae.eval_epoch(val_loader, reg_str, concord=concord_str)
		losses_history["train_loss"].append(tr_loss)
		losses_history["val_loss"].append(val_loss)

		colors = ['\033[91m', '\033[95m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
		for i, key in enumerate(tr_loss):
			color = colors[i % len(colors)]
			reset = '\033[0m'
			print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")

	model_path = "moe_vae_model.pth"
	torch.save(moe_vae.state_dict(), model_path)
	print(f"Trained MOE VAE model saved to {model_path}")

	with open("losses_history.json", "w") as f:
		json.dump(losses_history, f)
	print("Losses history saved to losses_history.json")