import glob
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch

from tqdm import tqdm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE

def get_data_path(origin:str, train:bool, round:int) -> str:
	return origin + ('tr/' if train else 'vl/') + f'round_{round}/'

class PNGDataset(Dataset):
	'''
	Custom Dataset for loading PNG images from a directory
	'''
	def __init__(self, path=None, images=None):
		if (path is not None and images is not None) or (path is None and images is None):
			raise ValueError("Provide either a path or images, not both.")
		
		self.from_disk = images is None

		self.files = glob.glob(path + 'img_*.png') if self.from_disk else []
		self.transform = torchvision.transforms.ToTensor()
		self.data = []
		self.images = images
		
	def __len__(self):
		return len(self.files) if self.from_disk else len(self.images)
	
	def __getitem__(self, idx):
		if self.from_disk:
			with Image.open(self.files[idx]) as im:
				img = im.convert('RGB')
			img = self.transform(img)
			return img
		else:
			img = self.images[idx]
			img = self.transform(img)
			return img

def make_image_dataloader_safe(data_dir:str, batch_size:int=256) -> DataLoader:
	print(f'Creating dataloaded from {data_dir}')
	dataset = PNGDataset(path=data_dir)
	dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
	return dataloader

class TrasitionDataset(Dataset):
	'''
	Custom Dataset for loading images-action
	'''
	def __init__(self, path:str, seq_len:int=10, vq:VQVAE=None):
		super().__init__()
		apr_path = path + '/action_reward_data.json'
		with open(apr_path, 'r') as f:
			apr_json = json.load(f)
			act = apr_json["actions"]
			prop = apr_json["proprioception"]
			rew = apr_json["reward"]
		latents = []
		to_tensor_ = torchvision.transforms.ToTensor()
		with torch.no_grad():
			print(f"Encoding dataset from {path} using VQ-VAE...")
			#for episode in tqdm(range(min(len(act), max_ep)), 'Encoding Dataset'):
			for episode in range(len(act)):
				latents.append([])
				for i in range(0, len(act[episode]) + 1, 64):
					imgs = []
					for j in range(i, min(i+64, len(act[episode])+1)):
						im_path = path + f"/img_{episode}_{j}.png"
						with Image.open(im_path) as im:
							img = im.convert('RGB')
							img = to_tensor_(img)
							imgs.append(img)
					imgs = torch.stack(imgs)
					imgs = imgs.to(vq.device)
					_, latent, _ = vq.quantize(vq.encode(imgs))
					latent = latent.detach().clone().cpu()
					for j in range(latent.shape[0]): # maybe there is a way to avoid this loop
						latents[-1].append(latent[j])

		self.representation = []
		self.actions = []
		self.proprioception = []
		self.reward = []
		print(f"Creating sequences of length {seq_len}...")
		#for episode in tqdm(range(min(len(act), max_ep)), 'Defining Dataset'):
		for episode in range(len(act)):
			for i in range(0, len(act[episode]) - seq_len + 1, 1):
				l = []
				p = []
				for j in range(seq_len+1):
					l.append(latents[episode][i+j].clone())# this because we have a list of tensors
					p.append(prop[episode][i+j][0:17]) # only take the first 17 proprioception values other are target positions
				lat = torch.stack(l)
				self.representation.append(lat)
				self.actions.append(act[episode][i:i+seq_len])
				self.proprioception.append(p)
				self.reward.append(rew[episode][i:i+seq_len])
				

	def __len__(self):
		return len(self.actions)
	
	def __getitem__(self, idx):
		return {
			'latent': self.representation[idx].detach(),
			'action': torch.tensor(self.actions[idx], dtype=torch.float32).detach(),
			'proprioception': torch.tensor(self.proprioception[idx], dtype=torch.float32).detach(),
			'reward': torch.tensor(self.reward[idx], dtype=torch.float32).detach()
		}

def make_seq_dataloader_safe(data_dir:str, vq:VQVAE, seq_len:int=10, batch_size:int=64) -> DataLoader:
	dataset = TrasitionDataset(path=data_dir, seq_len=seq_len, vq=vq)
	dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
	return dataloader