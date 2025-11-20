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
	def __init__(self, path=None, images=None):
		if (path is not None and images is not None) or (path is None and images is None):
			raise ValueError("Provide either a path or images, not both.")
		
		self.from_disk = images is None

		self.files = glob.glob(path + '/*.png') if self.from_disk else []
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
	shuffle: whether to shuffle the data
	returns: DataLoader
	'''
	if data_dir is None and images is None:
		raise ValueError("Provide either a data_dir or images.")
	print(f"Creating dataloader from {'images' if images is not None else data_dir}")
	dataset = PNGDataset(path=data_dir, images=images)
	train_size = int((1 - test_split) * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	return train_loader, val_loader

class PNGMultiViewDataset(Dataset):
	'''
	Custom Dataset for loading multi-view PNG images from a directory
	'''
	def __init__(self, path):
		self.view1_files = sorted(glob.glob(path + '/above_img_*.png'))
		self.view2_files = sorted(glob.glob(path + '/side_img_*.png'))
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