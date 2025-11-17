import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

basic = True
if basic:
	from vae.myVae import CVAE as VAE
else:
	from vae.vqVae import VQVAE as VAE
from vae.abstractVAE import trainVAE
from helpers.general import best_device
from helpers.data import make_img_dataloader
from global_var import CURRENT_ENV

LATENT_DIM = 32
REG_STRENGTH = 1.0
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()


if __name__ == "__main__":
	if basic:
		vae = VAE(latent_dim=LATENT_DIM, device=DEVICE)
	else:
		vae = VAE(codebook_size=512,
				code_depth=64,
				latent_dim=8,
				commitment_cost=0.25,
				device=DEVICE)
	print(vae)
	num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
	print(f"Number of trainable parameters: {num_params}")
	train_loader, val_loader = make_img_dataloader(data_dir=DATA_PATH, batch_size=64, test_split=0.2)
	trained_vae, loss_history = trainVAE(vae,
										train_loader,
										val_loader,
										num_epochs=NUM_EPOCHS,
										learning_rate=LEARNING_RATE,
										regularization_strength=REG_STRENGTH)
	
	model_path = "vae_model.pth"
	torch.save(trained_vae.state_dict(), model_path)
	print(f"Trained VAE model saved to {model_path}")
	
	import matplotlib.pyplot as plt

	for key in loss_history['train_loss']:
		plt.figure(figsize=(10, 6))
		plt.plot(loss_history['train_loss'][key], label='Train Loss')
		plt.plot(loss_history['val_loss'][key], label='Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title(f'VAE Training Loss - {key}')
		plt.legend()
		plt.grid(True)
		plt.savefig(f'vae_loss_history_{key}.png')
		plt.show()
	