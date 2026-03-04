import os
import sys

import torch
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device

from vae.vqVae import VQVAE
from global_var import CURRENT_ENV, LATENT_DIM, CODE_DEPTH, CODEBOOK_SIZE, SMOOTH
from helpers.data import make_image_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae
import matplotlib.pyplot as plt


if __name__ == "__main__":
	device = best_device()
	vq_vae = VQVAE(
		codebook_size=CODEBOOK_SIZE,
		code_depth=CODE_DEPTH,
		latent_dim=LATENT_DIM,
		commitment_cost=0.25,
		device=device,
		ema_mode=True,
	).to(device)
	print(f"Testing {CURRENT_ENV['env_name']} VAE model")
	vq_vae = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, True if SMOOTH > 0 else False, device)
	vq_vae.eval()

	test_loader = make_image_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], False, 0))

	dataiter = iter(test_loader)
	images = next(dataiter).to(device)
	
	with torch.no_grad():
		vq_images, _, _ = vq_vae.forward(images)
		vq_images = vq_images.cpu()
		images = images.cpu()

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
		plt.imshow(vq_images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Flatten VQ Reconstructed Images')

		# VQ reconstructed image
		plt.subplot(3, num_images, i + 1 + 2 * num_images)
		plt.imshow(vq_images[i].permute(1, 2, 0).numpy())
		plt.axis('off')
		if i == 0:
			plt.title('Flatten VQ Reconstructed Images')
	
	plt.savefig('reconstructed_images.png', dpi=600)

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
	sorted_indexes = sorted(indexes_array, reverse=True)
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(sorted_indexes)), sorted_indexes)
	plt.xlabel('Codebook Index (sorted)')
	plt.ylabel('Frequency')
	plt.title('Codebook Usage Frequencies')
	plt.savefig('codebook_usage_.png', dpi=600)
	#plt.show()
