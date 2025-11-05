import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torchvision.transforms as T
from PIL import Image

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from models.vae import VAE
from models.mdnrnn import MDNRNN, sample_mdn
from global_var import CURRENT_ENV

# The only thing that changes between PseudoDreamEnv and DreamEnv is that
# PseudoDreamEnv uses the real environment to get the reward and done signal
# while DreamEnv uses only the world model to simulate everything
# so only the step and render functions changes
# (this could be refactored to avoid code duplication, but for clarity we keep them separate)
# TODO: a nice idea can be a boolean flag in the init function to switch between the two modes

class DreamEnv(gym.Env):
	"""
	Completely simulated environment using the VAE and MDRNN models
	The starting state is obtained from the real environment
	Then the environment is simulated using only the world model
	"""


	def __init__(self, env_dict, temperature=1.0, render_mode="none"):
		super(DreamEnv, self).__init__()
		# Load the VAE and MDRNN models
		self.vae = VAE(
			latent_dim=env_dict['z_size'],
		)
		self.vae.load_state_dict(torch.load(env_dict['vae_model'], map_location=torch.device('cpu')))
		self.vae.eval()
		self.mdrnn = MDNRNN(
			z_size=env_dict['z_size'],
			a_size=env_dict['a_size'],
			rnn_size=env_dict['rnn_size'],
			n_gaussians=env_dict['num_gaussians'],
		)
		self.mdrnn.load_state_dict(torch.load(env_dict['mdrnn_model'], map_location=torch.device('cpu')))
		self.mdrnn.eval()
		self.temperature = temperature
		self.hidden_state = None
		self.current_mu = None
		# Initialize the environment
		self.render_mode = render_mode
		self.env_dict = env_dict
		if env_dict['special_call'] is not None:
			env_dict['special_call']()
		if env_dict['default_camera_config'] is not None:
			self.env = gym.make(env_dict['env_name'], render_mode='rgb_array', default_camera_config=env_dict['default_camera_config'])
		else:
			self.env = gym.make(env_dict['env_name'], render_mode='rgb_array')
		_, _ = self.env.reset()
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.mdrnn.z_size + self.mdrnn.rnn_size,), dtype=np.float32
		)
		self.step_count = 0

	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		seed: random seed
		options: additional options
		returns: initial observation (np.array) obtained encoding the first image and the initial hidden state
		'''
		super().reset(seed=seed, options=options)
		obs, _ = self.env.reset()
		img = self.env.render()
		img = Image.fromarray(img).resize((64, 64))
		img = T.ToTensor()(img).unsqueeze(0)
		with torch.no_grad():
			mu, _ = self.vae.encode(img)
			self.hidden_state = (torch.zeros(1, 1, self.mdrnn.rnn_size),
			                     torch.zeros(1, 1, self.mdrnn.rnn_size))
			representation = torch.cat([mu.squeeze(0), self.hidden_state[0].squeeze(0).squeeze(0)], dim=-1).numpy()
		self.step_count = 0
		self.current_mu = mu.squeeze(0)
		return representation, {}  # empty info dict

	def step(self, action,) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		with torch.no_grad():
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			mu, log_std, pi, self.hidden_state, reward, _ = self.mdrnn(self.current_mu.unsqueeze(0).unsqueeze(0), action_tensor, self.hidden_state)
			self.current_mu = sample_mdn(mu[0, 0, :, :], log_std[0, 0, :, :], pi[0, 0, :], self.temperature)
			self.step_count += 1
			mu = self.current_mu.clone()
			representation = torch.cat([mu, self.hidden_state[0].squeeze(0).squeeze(0)], dim=-1).numpy()
			terminated = self.step_count >= 1000
		return (
			representation, # based on world model
			reward.item(), # from world model
			terminated, # For now only based on step count
			False, # Truncated
			{} # empty dict
		)
	
	def render(self):
		if self.render_mode == "rgb_array":
			with torch.no_grad():
				img = self.vae.decode(self.current_mu.unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
				return img
		elif self.render_mode == "human":
			with torch.no_grad():
				img = self.vae.decode(self.current_mu.unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
				img = (img * 255).astype(np.uint8)
				image = Image.fromarray(img)
				image_resized = image.resize((256, 256))
				cv2.imshow('DreamEnv', np.array(image_resized))
				cv2.waitKey(100)
				return img
		elif self.render_mode == "dream":
			# decode the current latent state to an image
			raise NotImplementedError("Dream rendering mode not yet implemented")
		
	def close(self):
		self.env.close()
		pass

if __name__ == "__main__":
	env = DreamEnv(CURRENT_ENV, temperature=1, render_mode="human")
	observation, info = env.reset()
	env.render()
	done = False
	total_reward = 0
	while not done:
		action = env.action_space.sample()  # random action
		action[1] = max(action[1], 0.5)  # accelerate
		action[2] = min(action[2], 0.3)  # low brake
		observation, reward, terminated, truncated, info = env.step(action)
		print(observation.shape)
		print(f"Reward: {reward}")
		print(f"Info: {info}")
		env.render()
		done = terminated or truncated
		total_reward += reward
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()