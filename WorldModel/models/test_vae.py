import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from vae import VAE
from dataset_func import make_dataloaders
import matplotlib.pyplot as plt

# file to test VAE training
# shows a couple of reconstructed images after training
if __name__ == "__main__":
	vae = VAE()
	print(f"Testing {CURRENT_ENV['env_name']} VAE model")
	vae_load_path = os.path.join(CURRENT_ENV['data_dir'], 'vae_model.pth')
	vae.load_state_dict(torch.load(vae_load_path, map_location='cpu'))
	vae.eval()

	_, test_loader = make_dataloaders(CURRENT_ENV['img_dir'])

	# get a batch of test images
	dataiter = iter(test_loader)
	images = next(dataiter)
	
	# reconstruct images using VAE
	with torch.no_grad():
		recon_images, _, _ = vae.forward(images)

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