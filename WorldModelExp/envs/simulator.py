import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3.ppo import PPO
from PIL import Image
from torch.utils.data import DataLoader

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER

class PusherDreamEnv(gym.Env):
	"""
	Completely simulated environment using the VAE and MDRNN models
	The starting state is obtained from the real environment
	Then the environment is simulated using only the world model
	"""


	def __init__(self, vq:VQVAE, lstm:LSTMQuantized, dataloader:DataLoader, init_len:int=1, ep_len:int=20):
		super(PusherDreamEnv, self).__init__()
		self.max_len = ep_len # this way the model will learn only 20 steps, hopefully in the end he will manage to merge his knowledge
		self.step_count = 0
		self.i_len = init_len

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

		self.data = dataloader

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
			print("[WARNING] You forgot to properly initialize the environment with the correct sequence length")
			print("[WARNING] Maybe init length should be just 1")
			print("[WARNING] Should you use all the data? or only the last one as initialization?")
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
			image_resized = image.resize((512, 512), Image.NEAREST)
			cv2.imshow('DreamEnv', np.array(image_resized))
			cv2.waitKey(100)
			return image_resized
			#return img
		
	def close(self):
		self.env.close()
		pass

if __name__ == "__main__":
	SMOOTH = True
	KL = True
	vq = load_vq_vae(PUSHER, 64, 32, 4, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), 1024, SMOOTH, True, KL)
	env = PusherDreamEnv(vq, lstm, 18, 5)
	observation, _ = env.reset()
	frames = []
	frames.append(env.render())
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(PUSHER['models'] + 'agent', env)
	while not done:
		#action = env.action_space.sample()  # random action
		action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		frames.append(env.render())
		done = terminated or truncated
		total_reward += reward
		step_count += 1
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()

	GIF_PATH = "output.gif"
	FRAME_DURATION_MS = 50
	# frames[0].save(
    #     GIF_PATH,
    #     save_all=True,
    #     append_images=frames[1:],
    #     loop=0,                    # 0 = loop forever
    #     duration=FRAME_DURATION_MS,
    # )