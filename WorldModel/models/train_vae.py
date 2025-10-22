import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from vae import VAE
from dataset_func import make_dataloaders



# This function contains the code to train the VAE model
def train_vae():
	train_loader, _ = make_dataloaders(CURRENT_ENV['img_dir'], batch_size=128)
	vae = VAE()
	vae.train_(dataloader=train_loader, 
		optimizer=torch.optim.Adam(vae.parameters(), lr=1e-3),
		epochs=10,
		device='cuda' if torch.cuda.is_available() else 'mps',
		kld_weight=1.0
		)
	return vae
	
if __name__ == "__main__":
	vae = train_vae()
	# save the model
	vae_save_path = os.path.join(CURRENT_ENV['data_dir'], 'vae_model.pth')
	torch.save(vae.state_dict(), vae_save_path)