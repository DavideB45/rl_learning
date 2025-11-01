import glob
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch

class PNGDataset(Dataset):
	'''
	Custom Dataset for loading PNG images from a directory
	'''
	def __init__(self, path, transform=None, preload=True, images=None):
		self.files = glob.glob(path + '/*.png') if images is None else []
		self.transform = transform
		self.preload = preload
		self.data = []

		if images is not None:
			_to_tensor = torchvision.transforms.ToTensor()
			self.preload = True
			for img in images:
				if self.transform:
					img = self.transform(img)
				else:
					img = _to_tensor(img)
				self.data.append(img)
		else:
			if self.preload:
				_to_tensor = torchvision.transforms.ToTensor()
				for fp in self.files:
					img = Image.open(fp).convert('RGB')
					if self.transform:
						img = self.transform(img)
					else:
						img = _to_tensor(img)
					self.data.append(img)

	def __len__(self):
		return len(self.data) if self.preload else len(self.files)

	def __getitem__(self, idx):
		if self.preload:
			return self.data[idx]
		# fallback: load on demand
		img = Image.open(self.files[idx]).convert('RGB')
		if self.transform:
			img = self.transform(img)
		else:
			img = torchvision.transforms.ToTensor()(img)
		return img
	

def make_dataloaders(data_dir, test_split=0.2, batch_size=64, images=None):
	'''
	Create PyTorch dataloaders for training and testing datasets
	data_dir: directory containing the dataset images
	test_split: fraction of data to use for testing
	batch_size: batch size for dataloaders
	images: optional list of PIL Images to use instead of loading from disk
	returns: train_loader, test_loader
	'''
	print(f"Creating dataloaders from images in {data_dir}")
	dataset = PNGDataset(path=data_dir, images=images)
	test_size = int(len(dataset) * test_split)
	train_size = len(dataset) - test_size
	generator = torch.Generator().manual_seed(42)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_loader, test_loader


class SequenceDataset(Dataset):
	'''
	Custom Dataset for loading sequences from .json
	'''
	def __init__(self, path, seq_len=10, transform=None, data_=None):
		self.file = path
		self.seq_len = seq_len
		self.transform = transform

		self.mu = []
		self.log_var = []
		self.action = []
		self.reward = []
		self.done = []
		data = json.load(open(self.file, 'r')) if data_ is None else data_
		print(f"Loading sequence dataset from {path} with {len(data)} episodes")
		for experience in data:
			for i in range(len(experience['mu']) - seq_len + 1):
				self.mu.append(experience['mu'][i:i+seq_len])
				self.log_var.append(experience['log_var'][i:i+seq_len])
				self.action.append(experience['action'][i:i+seq_len])
				self.reward.append(experience['reward'][i:i+seq_len])
				self.done.append(experience['last_state'][i:i+seq_len])
		print(f"Converting to tensors")
		for i in range(len(self.mu)):
			self.mu[i] = torch.tensor(self.mu[i], dtype=torch.float32)
			self.log_var[i] = torch.tensor(self.log_var[i], dtype=torch.float32)
			self.action[i] = torch.tensor(self.action[i], dtype=torch.float32)
			self.reward[i] = torch.tensor(self.reward[i], dtype=torch.float32)
			self.done[i] = torch.tensor(self.done[i], dtype=torch.float32)


	def __len__(self):
		return len(self.mu)

	def __getitem__(self, idx):
		return {
			'mu': self.mu[idx],
			'log_var': self.log_var[idx],
			'action': self.action[idx],
			'reward': self.reward[idx],
			'done': self.done[idx]
		}
	
def make_sequence_dataloaders(data_file, seq_len=10, test_split=0.2, batch_size=64, data_=None):
	'''
	Create PyTorch dataloaders for sequence datasets
	data_file: path to the .json file containing the dataset
	seq_len: length of each sequence
	test_split: fraction of data to use for testing
	batch_size: batch size for dataloaders
	returns: train_loader, test_loader
	'''
	print(f"Creating sequence dataloaders from {data_file}")
	dataset = SequenceDataset(path=data_file, seq_len=seq_len, data_=data_)
	test_size = int(len(dataset) * test_split)
	train_size = len(dataset) - test_size
	generator = torch.Generator().manual_seed(42)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_loader, test_loader
