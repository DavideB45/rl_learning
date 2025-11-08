import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from general_func import best_device

try:
	from vae import VAE
except ImportError:
	from models.vae import VAE
from dataset_func import make_dataloaders



# This function contains the code to train the VAE model
def train_vae(vae:VAE, path:str=None, images=None, epochs:int=10) -> VAE:
	train_loader, val_loader = make_dataloaders(path, batch_size=64, images=images)
	print(f"Training {CURRENT_ENV['env_name']} VAE model")
	vae.train_(trloader=train_loader, vlloader=val_loader,
		optimizer=torch.optim.Adam(vae.parameters(), lr=1e-4),
		epochs=epochs,
		device=best_device(),
		kld_tolerance=0.5
		)
	return vae
	
if __name__ == "__main__":
	vae = VAE(
		latent_dim=CURRENT_ENV['z_size'],
	)
	vae = train_vae(vae=vae, path=CURRENT_ENV['img_dir'], images=None)
	# save the model
	torch.save(vae.state_dict(), CURRENT_ENV['vae_model'])