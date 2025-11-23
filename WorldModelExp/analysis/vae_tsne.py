import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from sklearn.manifold import TSNE
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))


from vae.myVae import CVAE
from vae.vqVae import VQVAE
from vae.moevae import MOEVAE
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
			model = load_vq_vae(CURRENT_ENV, codebook_size, code_depth, latent_dim, device)
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
					# trying to do 1D representation for TSNE
					#print(mu.shape)
					mu = mu.permute(0, 2, 3, 1).contiguous()
					#print(mu.shape)
					mu = mu.view(mu.size(0), -1)
					latent_representations.append(mu.cpu().squeeze(0).numpy())
					#print(mu.shape)
					#emb = model.quantizer.embedding
					# try to see the closest embeddings
					#emb_dist = torch.cdist(mu[:, 0:8], emb.weight.unsqueeze(0))
					#closest_emb_idxs = torch.argmin(emb_dist, dim=-1)
					#print(closest_emb_idxs)
					#print(emb.weight[closest_emb_idxs])
					#print(mu[:, 0:8])
		latent_representations = np.array(latent_representations)

		tsne = TSNE(n_components=2, random_state=42)
		latent_2d = tsne.fit_transform(latent_representations)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
		
		# TSNE plot
		scatter = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], c=range(len(images)), cmap='viridis')
		ax1.plot(latent_2d[:, 0], latent_2d[:, 1], color='gray', linewidth=0.5, alpha=0.5)
		plt.colorbar(scatter, ax=ax1, label='Time Step')
		ax1.set_title(f'TSNE of Latent Representations ({model_type.upper()} VAE)')
		ax1.set_xlabel('TSNE Dimension 1')
		ax1.set_ylabel('TSNE Dimension 2')
		
		# MSE plot
		mse_values = []
		for i in range(1, len(latent_representations)):
			mse = np.mean((latent_representations[i] - latent_representations[i-1])**2)
			mse /= latent_representations.shape[1]
			mse_values.append(mse)
		ax2.plot(range(1, len(images)), mse_values, marker='o')
		ax2.set_title(f'MSE Between Consecutive Latent Representations ({model_type.upper()} VAE)')
		ax2.set_xlabel('Time Step')
		ax2.set_ylabel('MSE')
		ax2.grid()
		
		plt.tight_layout()
		plt.show()