import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV

try:
	from vae import VAE
except ImportError:
	from models.vae import VAE
from dataset_func import make_dataloaders



# This function contains the code to train the VAE model
def train_vae(vae_:VAE=None, path:str=None, images=None, epochs:int=10) -> VAE:
	train_loader, _ = make_dataloaders(path, batch_size=100, images=images)
	vae = VAE() if vae_ is None else vae_
	print(f"Training {CURRENT_ENV['env_name']} VAE model")
	vae.train_(dataloader=train_loader, 
		optimizer=torch.optim.Adam(vae.parameters(), lr=1e-4),
		epochs=epochs,
		device='cuda' if torch.cuda.is_available() else 'mps',
		kld_tolerance=0.5
		)
	return vae
	
if __name__ == "__main__":
	vae = train_vae(path=CURRENT_ENV['img_dir'], images=None)
	# save the model
	vae_save_path = os.path.join(CURRENT_ENV['data_dir'], 'vae_model.pth')
	torch.save(vae.state_dict(), vae_save_path)