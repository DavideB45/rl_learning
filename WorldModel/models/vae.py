from torch import nn
import torch

# The default values are copied from the World Model paper
# but can be changed for more complex environments or to lower computation
class VAE(nn.Module):
	'''
	Convolutional Variational Autoencoder (VAE) Model
	'''
	def __init__(self, input_dim=64, latent_dim=32, conv_depth=4):
		'''
		input_dim: input image width/height (assumed square)
		latent_dim: dimension of latent space
		conv_depth: number of convolutional layers
		'''
		if input_dim != 64 or conv_depth != 4:
			raise NotImplementedError("Only the default architecture is implemented")
		super(VAE, self).__init__()
		# encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0),  # 31x31x32
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # 14x14x64
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0), # 6x6x128
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0), # 2x2x256
			nn.ReLU(),
			nn.Flatten() # 1024
		)
		self.fc_mu = nn.Linear(1024, latent_dim)
		self.fc_logvar = nn.Linear(1024, latent_dim)
		# decoder
		self.decoder_input = nn.Linear(latent_dim, 1024)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2, padding=0), # 5x5x128
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0), # 13x13x64
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0), # 30x30x32
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0), # 64x64x3
			nn.Sigmoid()
		)

	def encode(self, x):
		'''
		Encode input image x into latent mean and log variance
		'''
		h = self.encoder(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar
	
	def reparameterize(self, mu, logvar):
		'''
		Reparameterization trick to sample from N(mu, var) from N(0,1)
		'''
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	
	def decode(self, z):
		'''
		Decode latent variable z into reconstructed image
		'''
		h = self.decoder_input(z)
		# TODO: This can be wrong
		h = h.view(-1, 1024, 1, 1)
		x_recon = self.decoder(h)
		return x_recon

	def forward(self, x):
		'''
		Forward pass through the VAE
		returns reconstructed image, mean, log variance
		'''
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		x_recon = self.decode(z)
		return x_recon, mu, logvar
	
	def loss(self, x, x_recon, mu, logvar, kld_tolerance=1.0):
		'''
		Compute VAE loss = reconstruction loss + KL divergence
		kld_tolerance: tolerance for the KL divergence term
		'''
		#recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
		recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='mean')
		kld_raw = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
		kld_loss = torch.max(kld_raw - 0.5, torch.tensor(0.0, device=kld_raw.device))
		return recon_loss + kld_loss

	def train_(self, dataloader, optimizer, epochs=10, kld_tolerance=1.0, device='cpu'):
		'''
		Train the VAE model
		dataloader: PyTorch dataloader for training data
		optimizer: optimizer for training
		epochs: number of training epochs
		kld_tolerance: tolerance for the KL divergence term
		device: device to run the training on
		'''
		self.to(device)
		self.train()
		for epoch in range(epochs):
			total_loss = 0
			for batch in dataloader:
				batch = batch.to(device)
				optimizer.zero_grad()
				x_recon, mu, logvar = self.forward(batch)
				loss = self.loss(batch, x_recon, mu, logvar, kld_tolerance)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")