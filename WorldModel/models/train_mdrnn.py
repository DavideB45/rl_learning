import os
import sys

import torch
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from models.mdnrnn import MDNRNN
from dataset_func import make_sequence_dataloaders


NUM_EPOCHS = 20
SEQUENCE_LENGTH = 100
REWARD_WEIGHT = 10.0
NOISE_SCALE = 0.8

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
		batch_size=64,
		seq_len=seq_len,
		data_=data_
	)
	print(f"Training {CURRENT_ENV['env_name']} MDRNN model")
	optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
	device = 'cuda' if torch.cuda.is_available() else 'mps'
	mdrnn.to(device)
	mdrnn.train()
	history = {'train_nll': [], 'val_nll': [], 'train_reward_loss': [], 'val_reward_loss': []}
	val_nll, val_reward_loss = validate(mdrnn, val_loader, device)
	best_val_loss = 1000
	for epoch in range(epochs):
		epoch_nll = 0.0
		epoch_reward_loss = 0.0
		num_batches = 0
		for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
			epoch_nll += nll.item()
			epoch_reward_loss += reward_loss.item()
			num_batches += 1
			del x, log_var, action, reward_target, mu, logvar, pi, h, reward, done_logits
		history['train_nll'].append(epoch_nll / num_batches)
		history['train_reward_loss'].append(epoch_reward_loss / num_batches)
		print(f"Train NLL: {epoch_nll/num_batches}, Train Reward Loss: {epoch_reward_loss/num_batches}")
		mdrnn.eval()
		with torch.no_grad():
			val_nll, val_reward_loss = validate(mdrnn, val_loader, device)
			history['val_nll'].append(val_nll)
			history['val_reward_loss'].append(val_reward_loss)
			print(f"Val NLL: {val_nll}, Val Reward Loss: {val_reward_loss}")
			if val_nll + val_reward_loss < best_val_loss:
				torch.save(mdrnn.state_dict(), 'best_mdrnn_model.pth')
				best_val_loss = val_nll + val_reward_loss
		mdrnn.train()
	return mdrnn

if __name__ == "__main__":
	mdrnn = MDNRNN(
		z_size=CURRENT_ENV['z_size'],
		a_size=CURRENT_ENV['a_size'],
		rnn_size=CURRENT_ENV['rnn_size'],
		n_gaussians=CURRENT_ENV['num_gaussians'],
		reward_weight=REWARD_WEIGHT
	)
	mdrnn = train_mdrnn(mdrnn=mdrnn, seq_len=SEQUENCE_LENGTH, epochs=NUM_EPOCHS, noise_scale=NOISE_SCALE)
	# save the model
	mdrnn_save_path = os.path.join(CURRENT_ENV['data_dir'], 'mdrnn_model.pth')
	torch.save(mdrnn.state_dict(), mdrnn_save_path)