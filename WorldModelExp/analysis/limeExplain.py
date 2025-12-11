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
	
	def lime_weight_mask(explanation, label):
		# Get superpixel weights
		weights = dict(explanation.local_exp[label])
		# The segmentation used by LIME
		segments = explanation.segments
		mask = np.ones_like(segments, dtype=float)
		for seg_id, w in weights.items():
			mask[segments == seg_id] = w
		return mask

	for i in range(len(images)):
		full_mask = None
		full_weight_mask = None
		all_masks = []
		for j in range(LATENT_DIM_VQ*LATENT_DIM_VQ):
			explainer = lime_image.LimeImageExplainer()
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
				hide_rest=True,
				min_weight=0.01
			)
			if full_mask is None:
				full_mask = mask1.clip(0,1)
				all_masks.append(mask1.clip(0,1))
				full_weight_mask = lime_weight_mask(explanation, explanation.top_labels[0])/LATENT_DIM_VQ/ LATENT_DIM_VQ
			else:
				full_mask = full_mask*2 + mask1.clip(0,1)
				all_masks.append(mask1.clip(0,1))
				full_weight_mask = full_weight_mask + lime_weight_mask(explanation, explanation.top_labels[0])/LATENT_DIM_VQ/ LATENT_DIM_VQ

			weight_mask = lime_weight_mask(explanation, explanation.top_labels[0])
			print(f"Weight mask stats for latent {j}: min {weight_mask.min()}, max {weight_mask.max()}, mean {weight_mask.mean()}")
			print(f"All weights for latent {j}: {np.unique(lime_weight_mask(explanation, explanation.top_labels[0])).__len__()}")
			print(f"Full weight mask stats for latent {j}: min {full_weight_mask.min()}, max {full_weight_mask.max()}, mean {full_weight_mask.mean()}")
			print(f"All weights for full weight mask: {np.unique(full_weight_mask).__len__()}")
			# Show the template and mask for this latent
			plt.figure(figsize=(8, 4))
			plt.subplot(2, 2, 1)
			plt.imshow(temp)
			plt.title(f"Latent {j} Explanation")
			plt.axis('off')
			plt.subplot(2, 2, 2)
			plt.imshow(mask1)
			plt.title(f"Latent {j} Mask")
			plt.axis('off')
			plt.subplot(2, 2, 3)
			plt.imshow(weight_mask, cmap='gray')
			plt.title(f"Latent {j} Weights")
			plt.axis('off')
			plt.subplot(2, 2, 4)
			plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
			plt.imshow(full_weight_mask, alpha=0.2, cmap='gray')
			plt.title(f"Latent {j} Weights Overlapped")
			plt.axis('off')
			plt.tight_layout()
			plt.show()
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
