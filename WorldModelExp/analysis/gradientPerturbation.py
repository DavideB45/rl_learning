import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device

from vae.vqVae import VQVAE
from global_var import CURRENT_ENV
from helpers.data import make_img_dataloader
from helpers.model_loader import load_base_vae, load_vq_vae

LATENT_DIM_VQ = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 128
EMA_MODE = True

if __name__ == "__main__":
	device = best_device()
	vq_vae = VQVAE(
		codebook_size=CODEBOOK_SIZE,
		code_depth=CODE_DEPTH,
		latent_dim=LATENT_DIM_VQ,
		commitment_cost=0.25,
		device=device,
		ema_mode=EMA_MODE,
	).to(device)
	print(f"Testing {CURRENT_ENV['env_name']} VAE model")
	vq_vae = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM_VQ, EMA_MODE, device)
	vq_vae.eval()

	_, test_loader = make_img_dataloader(CURRENT_ENV['img_dir'])

	# get a batch of test images
	dataiter = iter(test_loader)
	images = next(dataiter).to(device)[0:3]
	
	def predict_grad(img:torch.Tensor, device=device, vq_vae=vq_vae):
		vq_vae.eval()
		img.requires_grad = True
		latents = vq_vae.encode(img)
		sim_err = 0
		for _ in range(10000):
			sim_latent = torch.randn_like(latents)
			sim_err += F.mse_loss(latents, sim_latent)
		sim_err.backward()
		return img.grad.data.cpu().numpy()

	for i in range(images.shape[0]):
		with torch.no_grad():
			reconstructed, _, _ = vq_vae.forward(images[i].unsqueeze(0))
		gradients = predict_grad(images[i].unsqueeze(0))
		gradients = gradients*gradients
		gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
		fig, axs = plt.subplots(1, 4, figsize=(12, 4))
		axs[0].imshow(transforms.ToPILImage()(images[i].cpu()))
		axs[0].set_title("Original Image")
		axs[1].imshow(transforms.ToPILImage()(reconstructed[0].cpu()))
		axs[1].set_title("Reconstructed Image")
		axs[2].imshow(np.transpose(gradients[0], (1, 2, 0)))
		axs[2].set_title("Gradient w.r.t Input")
		# Overlay gradients on original image
		overlay = np.array(transforms.ToPILImage()(images[i].cpu())).astype(np.float32) / 255.0
		grad_overlay = np.transpose(gradients[0], (1, 2, 0))
		overlay = 0.5 * overlay + 0.5 * grad_overlay
		overlay = np.clip(overlay, 0, 1)
		axs[3].imshow(overlay)
		axs[3].set_title("Overlay")
		plt.show()
