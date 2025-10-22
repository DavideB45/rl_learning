import os
import sys
from tkinter import Image

import numpy as np
import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import PENDULUM
from vae import VAE
from torch.utils.data import DataLoader, random_split
import glob
from PIL import Image
from torch.utils.data import Dataset

class PNGDataset(Dataset):
	def __init__(self, path, transform=None):
		self.files = glob.glob(path + '/*.png')
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		img = Image.open(self.files[idx]).convert('RGB')
		if self.transform:
			img = self.transform(img)
		return img


def make_dataloaders(data_dir, test_split=0.2, batch_size=64):
	'''
	Create PyTorch dataloaders for training and testing datasets
	data_dir: directory containing the dataset images
	test_split: fraction of data to use for testing
	batch_size: batch size for dataloaders
	returns: train_loader, test_loader
	'''
	dataset = PNGDataset(path=data_dir, transform=torchvision.transforms.Compose([
		torchvision.transforms.Resize((64, 64)),
		torchvision.transforms.ToTensor(),
	]))
	test_size = int(len(dataset) * test_split)
	train_size = len(dataset) - test_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_loader, test_loader

# This function contains the code to train the VAE model
def train_vae():
	train_loader, _ = make_dataloaders(PENDULUM['img_dir'])
	vae = VAE()
	vae.train_(dataloader=train_loader, 
		optimizer=torch.optim.Adam(vae.parameters(), lr=1e-3),
		epochs=20,
		device='cuda' if torch.cuda.is_available() else 'mps'
		)
	return vae
	
if __name__ == "__main__":
	vae = train_vae()
	# save the model
	vae_save_path = os.path.join(PENDULUM['data_dir'], 'vae_model.pth')
	torch.save(vae.state_dict(), vae_save_path)