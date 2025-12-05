import os
import sys

from matplotlib import transforms
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
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
	
	# encode the images to latent space
	with torch.no_grad():
		latents = vq_vae.encode(images)
		latents = latents.permute(0, 2, 3, 1).contiguous()
		latents = latents.view(latents.size(0), -1)
		latents = latents.cpu().squeeze(0).numpy()

	# use LIME to explain the latent representation
	explainer = lime_image.LimeImageExplainer()
	def predict(img, device=device, vq_vae=vq_vae):
		vq_vae.eval()
		img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
		#TODO: check and decide what is the img input
		with torch.no_grad():
			latents = vq_vae.encode(img_tensor)
			latents = latents.permute(0, 2, 3, 1).contiguous()
			latents = latents.view(latents.size(0), -1)
			latents = latents.cpu().squeeze(0).numpy()
		return latents
	
	