import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.myVae import CVAE as VAE
from vae.abstractVAE import trainVAE
from helpers.general import best_device
from helpers.data import make_img_dataloader
from global_var import CURRENT_ENV

LATENT_DIM = 32
REG_STRENGTH = 1.0
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()


if __name__ == "__main__":
	vae = VAE(latent_dim=LATENT_DIM, device=DEVICE)
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

	plt.figure(figsize=(10, 6))
	plt.plot(loss_history['train_loss']['total'], label='Train Loss')
	plt.plot(loss_history['val_loss']['total'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('VAE Training Loss')
	plt.legend()
	plt.grid(True)
	plt.savefig('vae_loss_history.png')
	plt.show()

	plt.figure(figsize=(10, 6))
	plt.plot(loss_history['train_loss']['kl'], label='Train KL Divergence')
	plt.plot(loss_history['val_loss']['kl'], label='Validation KL Divergence')
	plt.xlabel('Epoch')
	plt.ylabel('KL Divergence')
	plt.title('VAE KL Divergence History')
	plt.legend()
	plt.grid(True)
	plt.savefig('vae_kl_history.png')
	plt.show()

	plt.figure(figsize=(10, 6))
	plt.plot(loss_history['train_loss']['reconstruction'], label='Train Reconstruction Loss')
	plt.plot(loss_history['val_loss']['reconstruction'], label='Validation Reconstruction Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Reconstruction Loss')
	plt.title('VAE Reconstruction Loss History')
	plt.legend()
	plt.grid(True)
	plt.savefig('vae_reconstruction_loss_history.png')
	plt.show()
	