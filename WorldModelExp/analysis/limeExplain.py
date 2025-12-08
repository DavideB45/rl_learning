import os
import sys

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

CODE_DEPTH = 8
LATENT_DIM_VQ = 4
CODEBOOK_SIZE = 256
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
			_, _, latents = vq_vae.quantize(latents)
			# (10, 16)
			#print(f"predict function called, latents shape: {latents.shape}")
		return latents[:, latent_to_check:latent_to_check+1]
	
	for i in range(len(images)):
		# print(f"image {i}, size: {images[i].shape}, max: {images[i].max()}, min: {images[i].min()}")
		# Convert tensor to numpy array with correct format for LIME
		# img_np = images[i].permute(1, 2, 0).cpu().numpy()
		# img_np = (img_np * 255).astype('uint8')
		# print(f"img_np shape: {img_np.shape}, max: {img_np.max()}, min: {img_np.min()}")
		full_mask = None
		for j in range(LATENT_DIM_VQ*LATENT_DIM_VQ):
			explanation = explainer.explain_instance(
				images[i].cpu().numpy().transpose(1, 2, 0),
				lambda img: predict(img, latent_to_check=j),
				labels=(0,),
				hide_color=0,
				num_samples=1000,
				num_features=64,
				batch_size=30
			)
			temp, mask = explanation.get_image_and_mask(
				label=0,
				positive_only=True,
				num_features=5,
				hide_rest=False,
			)
			if full_mask is None:
				full_mask = mask / (LATENT_DIM_VQ*LATENT_DIM_VQ)
			else:
				full_mask = full_mask + (mask / (LATENT_DIM_VQ*LATENT_DIM_VQ))
		# Get reconstructed image
		with torch.no_grad():
			reconstructed, _, _ = vq_vae.forward(images[i].unsqueeze(0))
		
		plt.figure(figsize=(16, 4))
		plt.subplot(1, 4, 1)
		plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
		plt.title(f"Image {i}")
		plt.axis('off')
		
		plt.subplot(1, 4, 2)
		plt.imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
		plt.title("Reconstructed")
		plt.axis('off')

		plt.subplot(1, 4, 3)
		plt.imshow(full_mask)
		plt.title("LIME Mask")
		plt.axis('off')
		
		plt.subplot(1, 4, 4)
		plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
		plt.imshow(full_mask, alpha=0.5, cmap='jet')
		plt.title(f"Image {i} Overlapped")
		plt.axis('off')
		
		plt.tight_layout()
		plt.show()
	