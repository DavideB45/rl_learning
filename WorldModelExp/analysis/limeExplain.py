import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from lime import lime_image
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
	images = next(dataiter).to(device)
	
	explainer = lime_image.LimeImageExplainer()
	def predict(img, device=device, vq_vae=vq_vae, latent_to_check=0):
		vq_vae.eval()
		img = torch.tensor(img).permute(0, 3, 1, 2).float().to(device)
		with torch.no_grad():
			latents = vq_vae.encode(img)
			probs = vq_vae.encode_probabilities(latents)
		return probs[:, :, latent_to_check%LATENT_DIM_VQ, latent_to_check//LATENT_DIM_VQ]
	
	def best_index(img, device=device, vq_vae=vq_vae, latent_to_check=0):
		vq_vae.eval()
		print(f"Shape of img in best_index: {img.shape}")
		img = img.unsqueeze(0)
		with torch.no_grad():
			latents = vq_vae.encode(img)
			probs = vq_vae.encode_probabilities(latents)

		return torch.argmax(probs[:, :, latent_to_check%LATENT_DIM_VQ, latent_to_check//LATENT_DIM_VQ], dim=1).cpu().numpy()
	
	for i in range(len(images)):
		full_mask = None
		all_masks = []
		for j in range(LATENT_DIM_VQ*LATENT_DIM_VQ):
			explanation = explainer.explain_instance(
				images[i].cpu().numpy().transpose(1, 2, 0),
				lambda img: predict(img, latent_to_check=j),
				labels=tuple(range(CODEBOOK_SIZE)),
				hide_color=0,
				top_labels=CODEBOOK_SIZE,
				num_samples=1000,
				batch_size=30
			)
			best = best_index(images[i], latent_to_check=j)
			print(f"Latent {j}, best index: {best}, expected index: {explanation.top_labels[0:5]}, length: {len(explanation.top_labels)}")
			temp, mask1 = explanation.get_image_and_mask(
				label=explanation.top_labels[0],
				positive_only=True,
				negative_only=False,
				num_features=1000,
				hide_rest=False,
				min_weight=0.01
			)
			if full_mask is None:
				full_mask = mask1.clip(0,1)
				all_masks.append(mask1.clip(0,1))
			else:
				full_mask = full_mask*2 + mask1.clip(0,1)
				all_masks.append(mask1.clip(0,1))
		# check if there are masks that are identical
		unique_masks = []
		for m in all_masks:
			if not any(np.array_equal(m, um) for um in unique_masks):
				unique_masks.append(m)
		print(f"Number of unique masks for image {i}: {len(unique_masks)} out of {len(all_masks)}")
		# Get reconstructed image
		with torch.no_grad():
			reconstructed, _, _ = vq_vae.forward(images[i].unsqueeze(0))
			
		plt.figure(figsize=(20, 4))
		plt.subplot(1, 5, 1)
		plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
		plt.title(f"Image {i}")
		plt.axis('off')
		
		plt.subplot(1, 5, 2)
		plt.imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
		plt.title("Reconstructed")
		plt.axis('off')

		plt.subplot(1, 5, 3)
		plt.imshow(full_mask)
		plt.title("LIME Mask")
		plt.axis('off')
		
		plt.subplot(1, 5, 4)
		plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
		plt.imshow(full_mask, alpha=0.5, cmap='jet')
		plt.title(f"Image {i} Overlapped")
		plt.axis('off')
		
		plt.subplot(1, 5, 5)
		plt.imshow(mask1)
		plt.title("Mask Only")
		plt.axis('off')
		
		plt.tight_layout()
		plt.show()
