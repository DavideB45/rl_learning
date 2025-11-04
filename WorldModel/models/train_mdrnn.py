import os
import sys

import torch
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from models.mdnrnn import MDNRNN
from dataset_func import make_sequence_dataloaders


NUM_EPOCHS = 20
SEQUENCE_LENGTH = 15
REWARD_WEIGHT = 5.0

def sample_x(mu, log_var, noise_scale=1.0):
	std = torch.exp(0.5 * log_var)
	eps = torch.randn_like(std)
	return mu + eps * std * noise_scale

def validate(mdrnn, val_loader, device):
	mdrnn.eval()
	total_nll = 0.0
	total_reward_loss = 0.0
	with torch.no_grad():
		for batch in val_loader:
			x = batch['mu'].to(device)
			log_var = batch['log_var'].to(device)
			reward_target = batch['reward'].to(device)
			x = sample_x(x, log_var)
			action = batch['action'].to(device)
			mu, logvar, pi, h, reward, done_logits = mdrnn(x, action)
			nll = mdrnn.neg_log_likelihood(x, mu, logvar, pi)
			total_nll += nll.item()
			total_reward_loss += mdrnn.reward_loss(reward, reward_target).item()
			del x, log_var, action, reward_target, mu, logvar, pi, h, reward, done_logits
	avg_nll = total_nll / len(val_loader)
	avg_reward_loss = total_reward_loss / len(val_loader)
	return avg_nll, avg_reward_loss

def train_mdrnn(mdrnn:MDNRNN, data_:dict=None, seq_len:int=10, epochs:int=30, noise_scale:float=1.0) -> MDNRNN:
	train_loader, val_loader = make_sequence_dataloaders(
		CURRENT_ENV['transitions'],
		batch_size=32,
		seq_len=seq_len,
		data_=data_
	)
	print(f"Training {CURRENT_ENV['env_name']} MDRNN model")
	optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
	device = 'cuda' if torch.cuda.is_available() else 'mps'
	mdrnn.to(device)
	mdrnn.train()
	val_nll, val_reward_loss = validate(mdrnn, val_loader, device)
	for epoch in tqdm(range(epochs), desc="Training MDRNN"):
		for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}, Validation NLL: {val_nll:.4f}, Validation Reward Loss: {val_reward_loss:.4f}", leave=False):
			x = batch['mu'].to(device)
			log_var = batch['log_var'].to(device)
			x = sample_x(x, log_var, noise_scale=noise_scale)
			action = batch['action'].to(device)
			reward_target = batch['reward'].to(device)
			#done_target = batch['done'].to(device)
			optimizer.zero_grad()
			mu, logvar, pi, h, reward, done_logits = mdrnn(x, action)
			nll = mdrnn.neg_log_likelihood(x[:, 1:, :], mu[:, :-1, :], logvar[:, :-1, :], pi[:, :-1, :])
			#done_loss = mdrnn.done_loss(done_logits, done_target)
			reward_loss = mdrnn.reward_loss(reward[:, :-1, :], reward_target[:, 1:])
			loss = nll + reward_loss # + done_loss
			loss.backward()
			torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), max_norm=1.0)
			optimizer.step()
			del x, log_var, action, reward_target, mu, logvar, pi, h, reward, done_logits

		with torch.no_grad():
			val_nll, val_reward_loss = validate(mdrnn, val_loader, device)
	return mdrnn

if __name__ == "__main__":
	mdrnn = MDNRNN(
		z_size=CURRENT_ENV['z_size'],
		a_size=CURRENT_ENV['a_size'],
		rnn_size=CURRENT_ENV['rnn_size'],
		n_gaussians=CURRENT_ENV['num_gaussians'],
		reward_weight=REWARD_WEIGHT
	)
	mdrnn = train_mdrnn(mdrnn=mdrnn, seq_len=SEQUENCE_LENGTH, epochs=NUM_EPOCHS)
	# save the model
	mdrnn_save_path = os.path.join(CURRENT_ENV['data_dir'], 'mdrnn_model.pth')
	torch.save(mdrnn.state_dict(), mdrnn_save_path)