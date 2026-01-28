import glob
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torch

from tqdm import tqdm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE

class PNGDataset(Dataset):
	'''
	Custom Dataset for loading PNG images from a directory
	'''
	def __init__(self, path=None, images=None):
		if (path is not None and images is not None) or (path is None and images is None):
			raise ValueError("Provide either a path or images, not both.")
		
		self.from_disk = images is None

		self.files = glob.glob(path + '/img_*.png') if self.from_disk else []
		self.transform = torchvision.transforms.ToTensor()
		self.data = []
		self.images = images
		
	def __len__(self):
		return len(self.files) if self.from_disk else len(self.images)
	
	def __getitem__(self, idx):
		if self.from_disk:
			img = Image.open(self.files[idx]).convert('RGB')
			img = self.transform(img)
			return img
		else:
			img = self.images[idx]
			img = self.transform(img)
			return img
		
def make_img_dataloader(data_dir=None, images=None, batch_size=64, test_split=0.2):
	'''
	Create PyTorch dataloader for a dataset of images
	data_dir: directory containing the dataset images
	images: optional list of PIL Images to use instead of loading from disk
	batch_size: batch size for dataloader
	returns: DataLoader
	'''
	if data_dir is None and images is None:
		raise ValueError("Provide either a data_dir or images.")
	print(f"Creating dataloader from {'images' if images is not None else data_dir}")
	dataset = PNGDataset(path=data_dir, images=images)
	n_total = len(dataset)
	train_size = int((1 - test_split) * n_total)
	train_indices = list(range(0, train_size))
	test_indices = list(range(train_size, n_total))
	train_dataset = Subset(dataset, train_indices)
	test_dataset = Subset(dataset, test_indices)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
	return train_loader, test_loader
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	return train_loader, val_loader

class PNGMultiViewDataset(Dataset):
	'''
	Custom Dataset for loading multi-view PNG images from a directory
	'''
	def __init__(self, path):
		self.view1_files = sorted(glob.glob(path + '/front_img_*.png'))
		self.view2_files = sorted(glob.glob(path + '/above_img_*.png'))
		if len(self.view1_files) != len(self.view2_files):
			raise ValueError("Number of images in view1 and view2 do not match.")
		self.transform = torchvision.transforms.ToTensor()

	def __len__(self):
		return len(self.view1_files)
	
	def __getitem__(self, idx):
		img1 = Image.open(self.view1_files[idx]).convert('RGB')
		img2 = Image.open(self.view2_files[idx]).convert('RGB')
		img1 = self.transform(img1)
		img2 = self.transform(img2)
		return img1, img2
	
def make_multi_view_dataloader(data_dir, batch_size=64, test_split=0.2):
	'''
	Create PyTorch dataloader for a multi-view dataset of images
	data_dir: directory containing the dataset images
	batch_size: batch size for dataloader
	shuffle: whether to shuffle the data
	returns: DataLoader
	'''
	print(f"Creating multi-view dataloader from {data_dir}")
	dataset = PNGMultiViewDataset(path=data_dir)
	train_size = int((1 - test_split) * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	return train_loader, val_loader

class TrasitionDataset(Dataset):
	'''
	Custom Dataset for loading images-action
	'''
	def __init__(self, path:str, seq_len:int=10, vq:VQVAE=None, max_ep:int=1000000):
		super().__init__()
		act = json.load(open(path + "/action_reward_data.json", 'r'))["actions"]
		latents = []
		to_tensor_ = torchvision.transforms.ToTensor()
		self.max_ep = max_ep
		with torch.no_grad():
			print(f"Encoding dataset from {path} using VQ-VAE...")
			for episode in tqdm(range(min(len(act), max_ep)), 'Encoding Dataset'):
			#for episode in range(min(len(act), max_ep)):
				latents.append([])
				for i in range(len(act[episode]) + 1):
					im_path = path + f"imgs/img_{episode}_{i}.png"
					img = Image.open(im_path).convert('RGB')
					img = to_tensor_(img).unsqueeze(0).to(vq.device)
					_, latent, _ = vq.quantize(vq.encode(img))
					latent = latent.detach().squeeze(0).clone().cpu()
					latents[-1].append(latent)

		self.representation = []
		self.actions = []
		print(f"Creating sequences of length {seq_len}...")
		for episode in tqdm(range(min(len(act), max_ep)), 'Defining Dataset'):
		#for episode in range(min(len(act), max_ep)):
			for i in range(0, len(act[episode]) - seq_len + 1, 1):
				l = []
				for j in range(seq_len+1):
					l.append(latents[episode][i+j].clone())
				lat = torch.stack(l)
				self.representation.append(lat)
				self.actions.append(act[episode][i:i+seq_len])

	def __len__(self):
		return min(len(self.actions), self.max_ep)
	
	def __getitem__(self, idx):
		return {
			'latent': self.representation[idx].detach(),
			'action': torch.tensor(self.actions[idx], dtype=torch.float32).detach()
		}

def make_sequence_dataloaders(path:str, vq:VQVAE ,seq_len:int=10, test_split:float=0.2, batch_size:int=64, max_ep:int=9999999) -> tuple[DataLoader, DataLoader]:
	print(f"Creating sequence dataloader using data from {path}")
	dataset = TrasitionDataset(path=path, seq_len=seq_len, vq=vq, max_ep=max_ep)
	n_total = len(dataset)
	train_size = n_total - int(n_total * test_split)
	train_indices = list(range(0, train_size))
	test_indices = list(range(train_size, n_total))
	train_dataset = Subset(dataset, train_indices)
	test_dataset = Subset(dataset, test_indices)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
	return train_loader, test_loader
	generator = torch.Generator().manual_seed(42)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_loader, test_loader
