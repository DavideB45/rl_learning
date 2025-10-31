import math
from torch import nn
import torch

LOGSQRT2PI = 0.5 * math.log(2.0 * math.pi)

class MDNRNN(nn.Module):
	def __init__(self, z_size=32, a_size=3, n_gaussians=5, rnn_size=256, done_pos_weight=1.0):
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
		self.fc_logstd = nn.Linear(rnn_size, n_gaussians * z_size)
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
		logstd = self.fc_logstd(rnn_out)  # (batch_size, seq_len, n_gaussians * z_size)
		pi = self.fc_pi(rnn_out)  # (batch_size, seq_len, n_gaussians)

		mu = mu.view(batch_size, seq_len, self.n_gaussians, self.z_size)
		logstd = logstd.view(batch_size, seq_len, self.n_gaussians, self.z_size)
		pi = nn.functional.softmax(pi, dim=-1)  # Apply softmax to get mixture weights

		reward = self.reward_pred(rnn_out)  # (batch_size, seq_len, 1)
		done_logits = self.done_pred(rnn_out)  # (batch_size, seq_len, 1)
		return mu, logstd, pi, h, reward, done_logits

	def neg_log_likelihood(self, x, mu, logstd, pi, mask=None):
		'''
		Compute the negative log-likelihood of x given the MDN parameters
		x: target latent vectors (batch_size, seq_len, z_size)
		mu: means of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
		logvar: log variances of Gaussian mixtures (batch_size, seq_len, n_gaussians, z_size)
		pi: mixture weights (batch_size, seq_len, n_gaussians)
		mask: optional mask to apply (batch_size, seq_len)
		Returns:
			Negative log-likelihood loss
		'''
		n_gaussians = mu.size(2)

		x = x.unsqueeze(2).expand(-1, -1, n_gaussians, -1)  # (batch_size, seq_len, n_gaussians, z_size)

		#logstd = torch.clamp(logstd, min=-2.0, max=2.0)  # Prevent numerical issues?¿
		var = torch.exp(2*logstd)
		log_prob = -0.5 * ((x - mu)**2 / (var + 1e-8)) - logstd - LOGSQRT2PI  # (batch_size, seq_len, n_gaussians)
		log_prob = torch.sum(log_prob, dim=-1)  # Sum over z_size dimension -> (batch_size, seq_len, n_gaussians)

		log_pi = torch.log(pi + 1e-8)  # Normalize log mixture weights
		log_prob = log_prob + log_pi  # (batch_size, seq_len, n_gaussians)

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
		if mask is not None:
			print(f"Warning: mask for done_loss is not implemented correctly")
		print("WARNING: this function is not written correctly, pos_weight needs to be handled differently")
		pos_weight = torch.tensor(self.done_pos_weight, dtype=done_logits.dtype, device=done_logits.device)
		loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
		loss = loss_fn(done_logits, done_targets).mean()
		return loss
	
	def reward_loss(self, reward_pred, reward_target, mask=None) -> torch.Tensor:
		'''
		Compute mean squared error loss for reward prediction
		reward_pred: predicted rewards (batch_size, seq_len, 1)
		reward_target: ground truth rewards (batch_size, seq_len, 1)
		mask: optional mask to apply (batch_size, seq_len, 1)
		Returns:
			Reward prediction loss
		'''
		loss_fn = nn.MSELoss(reduction='none')
		reward_target = reward_target.unsqueeze(-1)  # Match dimensions
		loss = loss_fn(reward_pred, reward_target)
		loss = loss.mean()
		return loss

def sample_mdn(z_mu, z_logstd, pi, temperature=1.0):
    """
    Sample a single latent state z from an MDN output with temperature scaling.
    Arguments:
        z_mu:        (n_gaussians, z_size) tensor of means
        z_logstd:    (n_gaussians, z_size) tensor of log standard deviations
        pi:          (n_gaussians,) tensor of mixture weights
        temperature: float. Higher=T more random, lower=T more deterministic
    Returns:
        z_sample:    (z_size,) tensor, sampled latent vector
    """
    # Scale mixture logits and recompute probabilities
    logits = torch.log(pi + 1e-8) / temperature
    pi_temp = torch.softmax(logits, dim=-1)
    # Sample which Gaussian
    cat = torch.distributions.Categorical(pi_temp)
    idx = cat.sample()
    # Temperature scales std; softmax for mixture, exp for std
    mu = z_mu[idx]
    std = torch.exp(z_logstd[idx]) * temperature
    normal = torch.distributions.Normal(mu, std)
    z = normal.sample()
    return z
