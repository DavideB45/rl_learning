import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch

class PNGDataset(Dataset):
	'''
	Custom Dataset for loading PNG images from a directory
	'''
	def __init__(self, path, transform=None, preload=True):
		self.files = glob.glob(path + '/*.png')
		self.transform = transform
		self.preload = preload
		self.data = []

		if self.preload:
			# load and preprocess all images once
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
	

def make_dataloaders(data_dir, test_split=0.2, batch_size=64):
	'''
	Create PyTorch dataloaders for training and testing datasets
	data_dir: directory containing the dataset images
	test_split: fraction of data to use for testing
	batch_size: batch size for dataloaders
	returns: train_loader, test_loader
	'''
	print(f"Creating dataloaders from images in {data_dir}")
	dataset = PNGDataset(path=data_dir, transform=torchvision.transforms.Compose([
		#torchvision.transforms.Resize((64, 64)),
		torchvision.transforms.ToTensor(),
	]))
	test_size = int(len(dataset) * test_split)
	train_size = len(dataset) - test_size
	generator = torch.Generator().manual_seed(42)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_loader, test_loader