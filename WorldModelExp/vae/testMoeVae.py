import torch
import json
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.moevae import MOEVAE
from helpers.general import best_device
from helpers.data import make_multi_view_dataloader

LATENT_DIM = 32

DATA_PATH = 'data/pusher/multi_img/'
DEVICE = best_device()

if __name__ == "__main__":
	moe_vae = MOEVAE(
		latent_dim=LATENT_DIM,
		device=DEVICE,
		learn_gating=False,
	)
	print(moe_vae)
	num_params = sum(p.numel() for p in moe_vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	_, val_loader = make_multi_view_dataloader(
		data_dir=DATA_PATH,
		batch_size=64,
		test_split=0.2
	)

	# load pre-trained model
	moe_vae.load_state_dict(torch.load("moe_vae_model.pth", map_location=DEVICE))
	moe_vae.eval()

	# Show reconstructions for a batch of validation data
	dataiter = iter(val_loader)
	data = next(dataiter)
	view1, view2 = data
	view1 = view1.to(DEVICE)
	view2 = view2.to(DEVICE)

	with torch.no_grad():
		recon1, recon2, mu, logvar = moe_vae(view1, view2)
	recon1 = recon1.cpu().detach()
	recon2 = recon2.cpu().detach()
	view1 = view1.cpu().detach()
	view2 = view2.cpu().detach()
	num_images = 4
	fig, axes = plt.subplots(num_images, 4, figsize=(12, 3 * num_images))
	for i in range(num_images):
		axes[i, 0].imshow(view1[i].permute(1, 2, 0))
		axes[i, 0].set_title("Original View 1")
		axes[i, 1].imshow(recon1[i].permute(1, 2, 0).clamp(0, 1))
		axes[i, 1].set_title("Reconstructed View 1")
		axes[i, 2].imshow(view2[i].permute(1, 2, 0))
		axes[i, 2].set_title("Original View 2")
		axes[i, 3].imshow(recon2[i].permute(1, 2, 0).clamp(0, 1))
		axes[i, 3].set_title("Reconstructed View 2")
		for j in range(4):
			axes[i, j].axis('off')
	plt.tight_layout()
	plt.show()