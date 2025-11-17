import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device

basic = False
if basic:
	from vae.myVae import CVAE as VAE
else:
	from vae.vqVae import VQVAE as VAE
from global_var import CURRENT_ENV
from helpers.data import make_img_dataloader
import matplotlib.pyplot as plt

# file to test VAE training
# shows a couple of reconstructed images after training
if __name__ == "__main__":
	device = best_device()
	if basic:
		vae = VAE(
			latent_dim=CURRENT_ENV['z_size'],
		).to(device)
	else:
		vae = VAE(
			codebook_size=512,
			code_depth=64,
			latent_dim=8,
			commitment_cost=0.25,
			device=device
		).to(device)
	print(f"Testing {CURRENT_ENV['env_name']} VAE model")
	vae.load_state_dict(torch.load('vae_model.pth', map_location=device))
	vae.eval()

	_, test_loader = make_img_dataloader(CURRENT_ENV['img_dir'])

	# get a batch of test images
	dataiter = iter(test_loader)
	images = next(dataiter).to(device)
	
	# reconstruct images using VAE
	with torch.no_grad():
		recon_images, _, _ = vae.forward(images)
	
	recon_images = recon_images.cpu()
	images = images.cpu()

	# pick a random image and show the sum of all pixel values
	random_idx = torch.randint(0, images.size(0), (1,)).item()
	print(f"Sum of pixel values for random original image {random_idx}: {images[random_idx].sum().item():.4f}")
	print(f"Sum of pixel values for random reconstructed image {random_idx}: {recon_images[random_idx].sum().item():.4f}")
	
	# plot original and reconstructed images
	num_images = 6
	plt.figure(figsize=(12, 4))
	for i in range(num_images):
		# original image
		plt.subplot(2, num_images, i + 1)
		plt.imshow(images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Original Images')
		
		# reconstructed image
		plt.subplot(2, num_images, i + 1 + num_images)
		plt.imshow(recon_images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Reconstructed Images')
	
	plt.show()