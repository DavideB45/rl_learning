import os
import sys

import torch
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device

from vae.myVae import CVAE
from vae.vqVae import VQVAE
from global_var import CURRENT_ENV
from helpers.data import make_img_dataloader
from helpers.model_loader import load_base_vae, load_vq_vae
import matplotlib.pyplot as plt

LATENT_DIM = 32
KL_WEIGHT = 0.5

LATENT_DIM_VQ = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 64
EMA_MODE = True
# file to test VAE training
# shows a couple of reconstructed images after training
if __name__ == "__main__":
	device = best_device()
	base_vae = CVAE(
		latent_dim=LATENT_DIM,
	).to(device)
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
	vq_vae2 = load_vq_vae(CURRENT_ENV, 128, CODE_DEPTH, LATENT_DIM_VQ, EMA_MODE, device)
	#base_vae = load_base_vae(CURRENT_ENV, LATENT_DIM, KL_WEIGHT, device)	
	vq_vae.eval()
	#base_vae.eval()

	test_loader, _ = make_img_dataloader(CURRENT_ENV['img_dir'])

	# get a batch of test images
	dataiter = iter(test_loader)
	images = next(dataiter).to(device)
	
	# reconstruct images using VAE
	with torch.no_grad():
		# vq reconstruction
		vq_images, _, _ = vq_vae.forward(images)
		recon_images, _, _ = vq_vae2.forward(images)
		# base reconstruction
		#mu, logvar = base_vae.encode(images)
		#recon_images = base_vae.decode(mu)
		
	vq_images = vq_images.cpu()
	recon_images = recon_images.cpu()
	images = images.cpu()

	# pick a random image and show the sum of all pixel values
	random_idx = torch.randint(0, images.size(0), (1,)).item()
	print(f"Sum of pixel values for random original image {random_idx}: {images[random_idx].sum().item():.4f}")
	print(f"Sum of pixel values for random vq reconstructed image {random_idx}: {vq_images[random_idx].sum().item():.4f}")
	print(f"Sum of pixel values for random base reconstructed image {random_idx}: {recon_images[random_idx].sum().item():.4f}")
	
	# plot original and reconstructed images
	num_images = 6
	plt.figure(figsize=(12, 6))
	for i in range(num_images):
		# original image
		plt.subplot(3, num_images, i + 1)
		plt.imshow(images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Original Images')
		
		# reconstructed image
		plt.subplot(3, num_images, i + 1 + num_images)
		plt.imshow(recon_images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Codebook 128 Reconstructed Images')

		# VQ reconstructed image
		plt.subplot(3, num_images, i + 1 + 2 * num_images)
		plt.imshow(vq_images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Codebook 64 Reconstructed Images')
	
	plt.savefig('reconstructed_images.png', dpi=600)
	#plt.show()

	exit(0)
	indexes_array = [0 for _ in range(vq_vae.codebook_size)]
	avg_error = 0.0
	for batch in tqdm(test_loader):
		batch = batch.to(device)
		with torch.no_grad():
			rec, _, indexes = vq_vae.forward(batch)
			avg_error += vq_vae.reconstruction_loss(batch, rec).item()
		for idx in indexes.cpu().numpy().flatten():
			indexes_array[idx] += 1
	avg_error /= len(test_loader)
	print(f"Average reconstruction error (MSE) on test set: {avg_error:.4f}")
	print("Codebook usage frequencies:")
	used = 0
	for i, count in enumerate(indexes_array):
		used += 1 if count > 0 else 0
	print(f"percentage: {used / len(indexes_array) * 100:.4f}%")
	# Plot codebook usage histogram
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(indexes_array)), indexes_array)
	plt.xlabel('Codebook Index')
	plt.ylabel('Frequency')
	plt.title('Codebook Usage Frequencies')
	plt.savefig('codebook_usage.png', dpi=600)
	#plt.show()
			