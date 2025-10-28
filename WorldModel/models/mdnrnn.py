from torch import nn
import torch

LOGSQRT2PI = torch.log(torch.tensor(2 * torch.pi))

class MDNRNN(nn.Module):
	def __init__(self, z_size, a_size, n_gaussians, rnn_size, done_pos_weight=1.0):
		'''
		MDN-RNN Model
		z_size: dimension of latent space input
		a_size: dimension of action space input
		n_gaussians: number of Gaussian mixtures
		rnn_size: number of RNN hidden units
		done_pos_weight: positive weight for done prediction loss (to handle class imbalance)
		'''
		super(MDNRNN, self).__init__()
		self.z_size = z_size
		self.a_size = a_size
		self.n_gaussians = n_gaussians
		self.rnn_size = rnn_size
		self.done_pos_weight = done_pos_weight

		self.rnn = nn.LSTM(input_size=z_size + a_size, hidden_size=rnn_size, num_layers=1, batch_first=True)
		self.fc_mu = nn.Linear(rnn_size, n_gaussians * z_size)
		self.fc_logvar = nn.Linear(rnn_size, n_gaussians * z_size)
		self.fc_pi = nn.Linear(rnn_size, n_gaussians)

		self.reward_pred = nn.Linear(rnn_size, 1)
		self.done_pred = nn.Linear(rnn_size, 1)

	def forward(self, x, a, h=None):
		'''
		Forward pass through the MDN-RNN
		x: input latent vectors (batch_size, seq_len, z_size)
		a: input actions (batch_size, seq_len, a_size)
		h: initial hidden state (optional)
		Returns:
			mu: means of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
			logvar: log variances of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
			pi: mixture weights (batch_size, seq_len, n_gaussians)
			h: final hidden state
			reward: predicted rewards (batch_size, seq_len, 1)
			done_logits: predicted done_logits (batch_size, seq_len, 1)
		'''
		batch_size, seq_len, _ = x.size()
		rnn_input = torch.cat([x, a], dim=-1)  # Concatenate along feature dimension

		if h is None:
			h = (torch.zeros(1, batch_size, self.rnn_size).to(x.device),
			     torch.zeros(1, batch_size, self.rnn_size).to(x.device))

		rnn_out, h = self.rnn(rnn_input, h)  # rnn_out: (batch_size, seq_len, rnn_size)

		mu = self.fc_mu(rnn_out)  # (batch_size, seq_len, n_gaussians * z_size)
		logvar = self.fc_logvar(rnn_out)  # (batch_size, seq_len, n_gaussians * z_size)
		pi = self.fc_pi(rnn_out)  # (batch_size, seq_len, n_gaussians)

		mu = mu.view(batch_size, seq_len, self.n_gaussians, self.z_size)
		logvar = logvar.view(batch_size, seq_len, self.n_gaussians, self.z_size)
		pi = nn.functional.softmax(pi, dim=-1)  # Apply softmax to get mixture weights

		reward = self.reward_pred(rnn_out)  # (batch_size, seq_len, 1)
		done_logits = self.done_pred(rnn_out)  # (batch_size, seq_len, 1)
		return mu, logvar, pi, h, reward, done_logits
	
	def neg_log_likelihood(self, x, mu, logvar, pi):
		'''
		Compute the negative log-likelihood of x given the MDN parameters
		x: target latent vectors (batch_size, seq_len, z_size)
		mu: means of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
		logvar: log variances of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
		pi: mixture weights (batch_size, seq_len, n_gaussians)
		Returns:
			Negative log-likelihood loss
		'''
		batch_size, seq_len, z_size = x.size()
		n_gaussians = mu.size(2)

		x = x.unsqueeze(2).expand(-1, -1, n_gaussians, -1)  # (batch_size, seq_len, n_gaussians, z_size)

		var = torch.exp(logvar)
		log_prob = -0.5 * (((x - mu) ** 2) / var) - logvar - LOGSQRT2PI  # (batch_size, seq_len, n_gaussians)
		log_prob = torch.sum(log_prob, dim=-1)  # Sum over z_size dimension -> (batch_size, seq_len, n_gaussians)

		log_prob += torch.log(pi + 1e-8)  # Add log mixture weights (don't multiply because we're using log)

		log_sum_exp = torch.logsumexp(log_prob, dim=-1)  # (batch_size, seq_len)

		nll = -log_sum_exp.mean()  # Mean negative log-likelihood over batch and sequence

		return nll
	
	def done_loss(self, done_logits, done_targets, mask=None):
		'''
		Compute binary cross-entropy loss for done prediction
		done_logits: predicted done logits (batch_size, seq_len, 1)
		done_targets: ground truth done values (batch_size, seq_len, 1)
		mask: optional mask to apply (batch_size, seq_len, 1)
		Returns:
			Done prediction loss
		'''
		loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.done_pos_weight, reduction='none')
		loss = loss_fn(done_logits, done_targets)
		loss = loss if mask is None else (loss * mask).sum() / mask.sum()
		return loss
	
	def reward_loss(self, reward_pred, reward_target, mask=None):
		'''
		Compute mean squared error loss for reward prediction
		reward_pred: predicted rewards (batch_size, seq_len, 1)
		reward_target: ground truth rewards (batch_size, seq_len, 1)
		mask: optional mask to apply (batch_size, seq_len, 1)
		Returns:
			Reward prediction loss
		'''
		loss_fn = nn.MSELoss(reduction='none')
		loss = loss_fn(reward_pred, reward_target)
		loss = loss if mask is None else (loss * mask).sum() / mask.sum()
		return loss