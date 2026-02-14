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

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.data import make_seq_dataloader_safe
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER

class PusherDreamEnv(gym.Env):
	"""
	Completely simulated environment using the VAE and MDRNN models
	The starting state is obtained from the real environment
	Then the environment is simulated using only the world model
	"""


	def __init__(self, vq:VQVAE=None, lstm:LSTMQuantized=None, sequence_length=10, max_ep=30):
		super(PusherDreamEnv, self).__init__()
		self.max_len = 100

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.lstm = lstm
		self.lstm.eval()
		self.hidden_state = None

		self.env = gym.make('Pusher-v5', 
			render_mode='rgb_array',
			default_camera_config=PUSHER['default_camera_config'],
		)
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.vq_dim + self.lstm.hidden_dim,), dtype=np.float32
		)
		self.step_count = 0

		self.data = make_seq_dataloader_safe(PUSHER['data_dir'], self.vq, seq_len=sequence_length, traininig=True, batch_size=1, max_ep=max_ep)

	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		seed: random seed
		options: additional options
		returns: initial observation (np.array) obtained encoding the first image and the initial hidden state
		'''
		super().reset(seed=seed, options=options)
		if seed is not None:
			print("[WARNING] I haven't implemented seed it's always random")
		init_data = self.data.dataset[np.random.randint(len(self.data.dataset))]
		with torch.no_grad():
			_, pred, prop, _, h = self.lstm.forward(init_data['latent'][:-1, :].unsqueeze(0).to(self.vq.device), init_data['action'].unsqueeze(0).to(self.vq.device), init_data['proprioception'].unsqueeze(0).to(self.vq.device), None)
		self.hidden_state = h
		self.current_latent = pred[:, -1, :, :, :]
		self.current_prop = prop[:, -1, :]
		representation = torch.cat([self.current_latent.flatten(), self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		self.step_count = 0
		# print(f'Latent shape: {self.current_latent.shape}')
		# print(f'Current prop shape: {self.current_prop.shape}')
		return representation, {}

	def step(self, action,) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		with torch.no_grad():
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			_, pred, prop, rew, h = self.lstm.forward(self.current_latent.unsqueeze(0).to(self.vq.device), action_tensor.to(self.vq.device), self.current_prop.unsqueeze(0).to(self.vq.device), self.hidden_state)
		self.step_count += 1
		self.hidden_state = h
		self.current_latent = pred[:, -1, :, :, :]
		self.current_prop = prop[:, -1, :]
		representation = torch.cat([self.current_latent.flatten(), self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		terminated = self.step_count >= self.max_len
		# print(f'Latent shape: {self.current_latent.shape}')
		# print(f'Current prop shape: {self.current_prop.shape}')
		return (
			representation, # based on world model
			rew[:, -1].item(), # from world model
			terminated, # For now only based on step count
			False, # Truncated
			{} # empty dict
		)
	
	def render(self):
		with torch.no_grad():
			img = self.vq.decode(self.current_latent).squeeze(0).permute(1, 2, 0).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			image = Image.fromarray(img)
			image_resized = image.resize((256, 256))
			cv2.imshow('DreamEnv', np.array(image_resized))
			cv2.waitKey(100)
			return img
		
	def close(self):
		self.env.close()
		pass

if __name__ == "__main__":
	SMOOTH = False
	KL = False
	vq = load_vq_vae(PUSHER, 64, 16, 4, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), 1024, SMOOTH, True, KL)
	env = PusherDreamEnv(vq, lstm, 18, 3)
	env.reset()
	env.render()
	done = False
	total_reward = 0
	step_count = 0
	while not done:
		action = env.action_space.sample()  # random action
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		env.render()
		done = terminated or truncated
		total_reward += reward
		step_count += 1
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()