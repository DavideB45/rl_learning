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
from global_var import VAE_MODEL, MDRNN_MODEL, CURRENT_ENV

class PseudoDreamEnv(gym.Env):
	"""
	"""


	def __init__(self, env_dict, render_mode="none"):
		super(PseudoDreamEnv, self).__init__()
		# Load the VAE and MDRNN models
		self.vae = VAE()
		self.vae.load_state_dict(torch.load(env_dict['data_dir'] + VAE_MODEL, map_location=torch.device('cpu')))
		self.mdrnn = MDNRNN()
		self.mdrnn.load_state_dict(torch.load(env_dict['data_dir'] + MDRNN_MODEL, map_location=torch.device('cpu')))
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
		self.hidden_state = None

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
		print(img.shape)
		with torch.no_grad():
			mu, _ = self.vae.encode(img)
			self.hidden_state = (torch.zeros(1, 1, self.mdrnn.rnn_size),
			                     torch.zeros(1, 1, self.mdrnn.rnn_size))
			representation = torch.cat([mu.squeeze(0), self.hidden_state[0].squeeze(0).squeeze(0)], dim=-1).numpy()

		return representation, {}  # empty info dict

	def step(self, action,) -> tuple:
		'''
		Step the environment using the MDRNN to get the hidden state
		And VAE to encode the image
		Apart from that, uses the real environment to get the reward and done signal
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		obs, reward, terminated, truncated, info = self.env.step(action)
		img = self.env.render()
		img = Image.fromarray(img).resize((64, 64))
		img = T.ToTensor()(img).unsqueeze(0)
		with torch.no_grad():
			mu, _ = self.vae.encode(img)
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			mu = mu.unsqueeze(0)
			_, _, _, self.hidden_state, _, _ = self.mdrnn(mu, action_tensor, self.hidden_state)
			representation = torch.cat([mu.squeeze(0).squeeze(0), self.hidden_state[0].squeeze(0).squeeze(0)], dim=-1).numpy()
		return (
			representation, # based on world model
			reward, # from real env (can be modified later)
			terminated, # from real env
			truncated, # from real env
			info # from real env (probably empty)
		)
	
	def render(self):
		if self.render_mode == "rgb_array":
			img = self.env.render()
			import matplotlib.pyplot as plt
			plt.imshow(img)
			plt.axis('off')
			plt.show()
			return img
		elif self.render_mode == "dream":
			# decode the current latent state to an image
			raise NotImplementedError("Dream rendering mode not yet implemented")
		
	def close(self):
		self.env.close()
		pass

if __name__ == "__main__":
	env = PseudoDreamEnv(CURRENT_ENV, render_mode="rgb_array")
	observation, info = env.reset()
	env.render()
	done = False
	while not done:
		action = env.action_space.sample()  # random action
		observation, reward, terminated, truncated, info = env.step(action)
		print(observation.shape)
		print(f"Reward: {reward}")
		print(f"Info: {info}")
		env.render()
		done = terminated or truncated
		if done:
			print(f"Game over! Reward: {reward}")
	env.close()