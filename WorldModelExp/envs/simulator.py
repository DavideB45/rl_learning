import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import torch
from stable_baselines3.ppo import PPO
from PIL import Image
from torch.utils.data import DataLoader
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import get_data_path, make_seq_dataloader_safe
from vae.vqVae import VQVAE
from dynamics.lstmc import LSTMQClass
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER

class PusherDreamEnv(VecEnv):
	"""
	Completely simulated environment using the VAE and MDRNN models
	The starting state is obtained from the real environment
	Then the environment is simulated using only the world model
	"""


	def __init__(self, vq:VQVAE, lstm:LSTMQClass, dataloader:DataLoader, init_len:int=1, ep_len:int=20, num_envs: int = 1):
		
		self.num_envs = num_envs
		self.max_len = ep_len # this way the model will learn only 20 steps, hopefully in the end he will manage to merge his knowledge
		self.step_count = 0
		self.i_len = init_len

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.lstm = lstm
		self.lstm.eval()
		self.hidden_state = None # (num_envs, hidden_dim)

		self.env = gym.make('Pusher-v5', 
			render_mode='rgb_array',
			default_camera_config=PUSHER['default_camera_config'],
		)
		self.single_action_space = self.env.action_space
		self.single_observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.vq_dim + self.lstm.hidden_dim,), dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, 
			shape=(num_envs, self.vq_dim + self.lstm.hidden_dim), 
			dtype=np.float32
		)
		self.action_space = spaces.Box(
			low=self.env.action_space.low[0], 
			high=self.env.action_space.high[0],
			shape=(num_envs, *self.env.action_space.shape),
			dtype=self.env.action_space.dtype
		)
		super(PusherDreamEnv, self).__init__(
			num_envs=num_envs,
			action_space=self.single_action_space,
			observation_space=self.single_observation_space	
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
		init_data_list = [self.data.dataset[np.random.randint(len(self.data.dataset))] for _ in range(self.num_envs)]
		with torch.no_grad():
			print("[WARNING] Maybe init length should be just 1")
			print("[WARNING] Should you use all the data? or only the last one as initialization?")
			latents = torch.stack([init_data['latent'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)
			actions = torch.stack([init_data['action'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)
			props = torch.stack([init_data['proprioception'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)

			_, pred, prop, _, h = self.lstm.forward(latents, actions, props, None)

			self.hidden_state = h
			self.current_latent = pred[:, -1, :, :, :]
			self.current_prop = prop[:, -1, :]
			latent_flat = self.current_latent.reshape(self.num_envs, -1)
			hidden_flat = self.hidden_state[0].reshape(self.num_envs, -1)

			representation = torch.cat([latent_flat, hidden_flat], dim=-1).cpu().numpy()
		#print(latent_flat.shape)
		#print(hidden_flat.shape)
		self.step_count = 0
		#print(f'Latent shape: {self.current_latent.shape}')
		#print(f'Current prop shape: {self.current_prop.shape}')
		#print(representation.shape)
		#exit()
		return representation

	def step(self, actions) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		if actions.ndim == 1:
			actions = actions[np.newaxis, :]
		with torch.no_grad():
			action_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(self.vq.device)
			latent_input = self.current_latent.unsqueeze(1).to(self.vq.device)
			prop_input = self.current_prop.unsqueeze(1).to(self.vq.device)
			_, pred, prop, rew, h = self.lstm.forward(latent_input, action_tensor, prop_input, self.hidden_state)
			
			self.step_count += 1
			self.hidden_state = h
			self.current_latent = pred[:, -1, :, :, :]
			self.current_prop = prop[:, -1, :]

			latent_flat = self.current_latent.reshape(self.num_envs, -1)
			hidden_flat = self.hidden_state[0].reshape(self.num_envs, -1)
			representation = torch.cat([latent_flat, hidden_flat], dim=-1).cpu().numpy()

			terminateds = np.array([self.step_count >= self.max_len] * self.num_envs, dtype=bool)
			truncateds = np.zeros(self.num_envs, dtype=bool)
		# print(f'Latent shape: {self.current_latent.shape}')
		# print(f'Current prop shape: {self.current_prop.shape}')
		return (
			representation, # based on world model
			rew[:, -1].item(), # from world model
			terminateds, # For now only based on step count
			truncateds, # Truncated
			{} # empty dict
		)
	
	def render(self):
		if(self.num_envs != 1):
			print('[WARNING] trying to render vectorized env, you are not Doctor strange')
			return
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
	env = PusherDreamEnv(vq=vq, lstm=lstm, dataloader=make_seq_dataloader_safe(get_data_path(PUSHER['data_dir'], True, 0), vq, 100, 1), 
					  num_envs=1, ep_len=100, init_len=1)
	observation, _ = env.reset()
	frames = []
	frames.append(env.render())
	done = False
	total_reward = 0
	step_count = 0
	#agent = PPO.load(PUSHER['models'] + 'agent', env)
	while not done:
		action = env.action_space.sample()  # random action
		#action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		frames.append(env.render())
		done = terminated or truncated
		total_reward += reward
		step_count += 1
		time.sleep(2)
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