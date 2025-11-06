import torch
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.mdnrnn import MDNRNN, sample_mdn
from models.vae import VAE

from environments.create_real import make_first_frame
from global_var import CURRENT_ENV

def show_dream_sample(mdrnn, vae, device, seq_len=100, temp=1.0):
	mdrnn.eval()
	vae.eval()
	with torch.no_grad():
		# Get the first frame from the real environment
		first_frame, action_space = make_first_frame(CURRENT_ENV)
		first_frame = transforms.ToTensor()(first_frame).unsqueeze(0).to(device)  # [1, C, H, W]
		mu, log_var = vae.encode(first_frame)
		#z = vae.reparameterize(mu, log_var)  # [1, z_size]
		z = mu  # Use mean as latent vector
		
		h = None  # Initial hidden state
		z_seq = []
		for t in range(seq_len):
			action = torch.zeros((1, action_space.shape[0])).to(device)  # Zero action
			if CURRENT_ENV['env_name'] == 'CarRacing-v3':
				action[0, 1] = 0.5  # constant gas
			mu, logvar, pi, h, _, _ = mdrnn(z.unsqueeze(1), action.unsqueeze(1), h)
			z = sample_mdn(mu[0, 0, :, :], logvar[0, 0, :, :], pi[0, 0, :], temperature=temp)
			z = z.unsqueeze(0)  # Add batch dimension
			z_seq.append(z.cpu())
		
		z_seq = torch.stack(z_seq, dim=1)  # [1, seq_len, z_size]
		recon_images = vae.decode(z_seq.view(-1, z_seq.size(-1))).view(1, seq_len, *first_frame.shape[1:])  # [1, seq_len, C, H, W]
		
		# Display the generated images
		import matplotlib.pyplot as plt
		num_images = min(10, seq_len)
		# keep only num_images for display at regular intervals
		#step = seq_len // num_images
		#recon_images = recon_images[:, ::3, :, :, :]
		plt.figure(figsize=(15, 3))
		for i in range(num_images):
			plt.subplot(1, num_images, i + 1)
			plt.imshow(recon_images[0, i].permute(1, 2, 0).cpu().numpy())
			plt.axis('off')
			if i == 0:
				plt.title('Dreamed Sequence')
		plt.show()

if __name__ == "__main__":
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	mdrnn = MDNRNN(
		z_size=CURRENT_ENV['z_size'],
		a_size=CURRENT_ENV['a_size'],
		rnn_size=CURRENT_ENV['rnn_size'],
		n_gaussians=CURRENT_ENV['num_gaussians'],
	).to(device)
	mdrnn.eval()
	vae = VAE(
		latent_dim=CURRENT_ENV['z_size'],
	).to(device)
	vae.eval()
	
	# Load pretrained models
	mdrnn.load_state_dict(torch.load(CURRENT_ENV['mdrnn_model'], map_location=device))
	vae.load_state_dict(torch.load(CURRENT_ENV['vae_model'], map_location=device))

	show_dream_sample(mdrnn, vae, device, seq_len=50, temp=0.8)