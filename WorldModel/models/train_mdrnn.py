import os
import sys

import torch
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from models.mdnrnn import MDNRNN
from dataset_func import make_sequence_dataloaders


NUM_EPOCHS = 30

def sample_x(mu, log_var):
	std = torch.exp(0.5 * log_var)
	eps = torch.randn_like(std)
	return mu + eps * std

def validate(mdrnn, val_loader, device):
	mdrnn.eval()
	total_nll = 0.0
	with torch.no_grad():
		for batch in val_loader:
			x = batch['mu'].to(device)
			log_var = batch['log_var'].to(device)
			x = sample_x(x, log_var)
			action = batch['action'].to(device)
			mu, logvar, pi, h, reward, done_logits = mdrnn(x, action)
			nll = mdrnn.neg_log_likelihood(x, mu, logvar, pi)
			total_nll += nll.item()
	avg_nll = total_nll / len(val_loader)
	return avg_nll

def train_mdrnn(mdrnn_:MDNRNN=None, data_:dict=None, seq_len:int=10, epochs:int=30) -> MDNRNN:
	train_loader, val_loader = make_sequence_dataloaders(
		CURRENT_ENV['transitions'],
		batch_size=100,
		seq_len=seq_len,
		data_=data_
	)
	mdrnn = MDNRNN() if mdrnn_ is None else mdrnn_
	print(f"Training {CURRENT_ENV['env_name']} MDRNN model")
	optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	mdrnn.to(device)
	mdrnn.train()
	for epoch in tqdm(range(epochs), desc="Training MDRNN"):
		for batch in train_loader:
			x = batch['mu'].to(device)
			log_var = batch['log_var'].to(device)
			#x = sample_x(x, log_var)
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
		print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item()}, NLL: {nll.item()}, Reward Loss: {reward_loss.item()}")
		val_nll = validate(mdrnn, val_loader, device)
		print(f"Validation NLL: {val_nll}")
	return mdrnn

if __name__ == "__main__":
	mdrnn = train_mdrnn()
	# save the model
	mdrnn_save_path = os.path.join(CURRENT_ENV['data_dir'], 'mdrnn_model.pth')
	torch.save(mdrnn.state_dict(), mdrnn_save_path)