import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from sklearn.manifold import TSNE
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))


from helpers.model_loader import *

from helpers.general import best_device
from global_var import CURRENT_ENV

def get_img(renderer, size=(64, 64)) -> Image.Image:
	renderer.camera_id = 2
	img = renderer.render(render_mode='rgb_array')
	img = Image.fromarray(img)
	img = img.resize(size)
	return img

def get_sequence() -> list[Image.Image]:
	env = gym.make(CURRENT_ENV['env_name'], 
				render_mode='rgb_array',
				default_camera_config=CURRENT_ENV['default_camera_config'],
				)
	renderer = env.env.env.env.mujoco_renderer
	images = []
	_, _ = env.reset(seed=42)
	images.append(get_img(renderer, size=(64, 64)))
	terminated, truncated = False, False
	while not (terminated or truncated):
		action = env.action_space.sample()
		_, _, terminated, truncated, info = env.step(action)
		images.append(get_img(renderer, size=(64, 64)))
	return images

def img_distance(img1: Image.Image, img2: Image.Image) -> float:
	arr1 = np.array(img1).astype(np.float32) / 255.0
	arr2 = np.array(img2).astype(np.float32) / 255.0
	return np.mean(np.abs(arr1 - arr2))

def latent_distance(latent1: np.ndarray, latent2: np.ndarray) -> float:
	return np.mean(np.abs(latent1 - latent2))

if __name__ == "__main__":
	device = best_device()
	
	images = get_sequence()
	show = True
	while show:
		model_type = input("Enter VAE model type (base/vq/moe): ")
		if model_type == 'base':
			latent_dim = int(input("Enter latent dimension size: "))
			kl_b = float(input("Enter KL beta value: "))
			model = load_base_vae(CURRENT_ENV, latent_dim, kl_b, device)
		elif model_type == 'vq':
			codebook_size = int(input("Enter codebook size: "))
			code_depth = int(input("Enter code depth: "))
			latent_dim = int(input("Enter latent dimension size: "))
			ema_mode = input("Use EMA mode? (y/n): ").lower() == 'y'
			model = load_vq_vae(CURRENT_ENV, codebook_size, code_depth, latent_dim, ema_mode=ema_mode, device=device)
		elif model_type == 'moe':
			latent_dim = int(input("Enter latent dimension size: "))
			kl_b = float(input("Enter KL beta value: "))
			concordance_reg = float(input("Enter concordance regularization value: "))
			model = load_moe_vae(CURRENT_ENV, latent_dim, kl_b, concordance_reg, device)
		else:
			print("Invalid model type")
			exit(1)

		model.eval()
		latent_representations = []
		with torch.no_grad():
			for img in images:
				img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
				if model_type == 'base':
					mu, _ = model.encode(img_tensor)
					latent_representations.append(mu.cpu().squeeze(0).numpy())
				elif model_type == 'moe':
					mu, _ = model.encode_expert1(img_tensor)
					latent_representations.append(mu.cpu().squeeze(0).numpy())
				elif model_type == 'vq':
					mu = model.encode(img_tensor)
					_, mu, _ = model.quantize(mu)
					mu = mu.permute(0, 2, 3, 1).contiguous()
					mu = mu.view(mu.size(0), -1)
					latent_representations.append(mu.cpu().squeeze(0).numpy())

		# Compute distance matrices
		num_images = len(images)
		img_dist_matrix = np.zeros((num_images, num_images))
		latent_dist_matrix = np.zeros((num_images, num_images))
		for i in range(num_images):
			for j in range(num_images):
				img_dist_matrix[i, j] = img_distance(images[i], images[j])
				latent_dist_matrix[i, j] = latent_distance(latent_representations[i], latent_representations[j])
		
		# Plot the distance matrices
		plt.figure(figsize=(12, 5))
		plt.subplot(1, 2, 1)
		plt.title("Image Distance Matrix")
		plt.imshow(img_dist_matrix, cmap='viridis')
		plt.colorbar()
		plt.subplot(1, 2, 2)
		plt.title("Latent Distance Matrix")
		plt.imshow(latent_dist_matrix, cmap='viridis')
		plt.colorbar()
		plt.show()
		# Correlation
		img_dists = img_dist_matrix.flatten()
		latent_dists = latent_dist_matrix.flatten()
		correlation = np.corrcoef(img_dists, latent_dists)
		print(f"Correlation between image distances and latent distances: {correlation[0, 1]:.4f}")
		print(f"full correlation matrix:\n{correlation}")
		print("Enter 'y' to analyze another model, any other key to exit.")
		cont = input("Analyze another model? (y/n): ").lower()
		if cont != 'y':
			show = False


