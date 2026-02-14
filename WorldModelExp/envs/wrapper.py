import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torchvision.transforms as T
import json
from PIL import Image
from stable_baselines3.ppo import PPO
from tqdm import tqdm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER

class PusherWrapEnv(gym.Env):
	"""
	This environemt is a wrapper of the real environment used at inference time
	Since the agent can't do inference directly on the data coming from the environment
	"""


	def __init__(self, vq:VQVAE=None, lstm:LSTMQuantized=None):
		super(PusherWrapEnv, self).__init__()

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.lstm = lstm
		self.lstm.eval()

		self.env = gym.make('Pusher-v5', 
			render_mode='rgb_array',
			default_camera_config=PUSHER['default_camera_config'],
		)
		self.renderer = self.env.env.env.env.mujoco_renderer
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.vq_dim + self.lstm.hidden_dim,), dtype=np.float32
		)
		self.to_tensor_ = T.ToTensor()

	def get_img(self) -> Image.Image:
		'''
		Renders the current frame of the environment and resizes it.
		Args:
			env: gym environment
			size: desired size of the image
		Returns:
			Image.Image: resized image
		'''
		self.renderer.camera_id = 2
		img = self.renderer.render(render_mode='rgb_array')
		img = Image.fromarray(img)
		img = img.resize((64, 64))
		return img
	
	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		seed: random seed
		options: additional options
		returns: initial observation (np.array) obtained encoding the first image and the initial hidden state
		'''
		super().reset(seed=seed, options=options)
		prop, _ = self.env.reset(seed=seed)
		img = self.get_img()
		with torch.no_grad():
			t_img = self.to_tensor_(img).unsqueeze(0).to(self.vq.device)
			_, lat, _ = self.vq.quantize(self.vq.encode(t_img))
			h = (torch.zeros(1, 1, self.lstm.hidden_dim).to(self.vq.device),
				 torch.zeros(1, 1, self.lstm.hidden_dim).to(self.vq.device))
		self.hidden_state = h
		self.current_latent = lat
		self.current_prop = torch.tensor(prop[:17], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		self.current_render = img
		representation = torch.cat([self.current_latent.flatten(), self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		# print(f'Latent shape: {self.current_latent.shape}')
		# print(f'Current prop shape: {self.current_prop.shape}')
		return representation, {}

	def step(self, action,) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		prop, reward, terminated, truncated, info = self.env.step(action)
		prop = prop[0:17]
		img = self.get_img()
		with torch.no_grad():
			t_img = self.to_tensor_(img).unsqueeze(0).to(self.vq.device)
			_, lat, _ = self.vq.quantize(self.vq.encode(t_img))
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			_, _, _, _, h = self.lstm.forward(self.current_latent.unsqueeze(0).to(self.vq.device), action_tensor.to(self.vq.device), self.current_prop.unsqueeze(0).to(self.vq.device), self.hidden_state)
		self.hidden_state = h
		self.current_latent = lat
		self.current_prop = torch.tensor(prop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		self.current_render = img
		representation = torch.cat([self.current_latent.flatten(), self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		# print(f'Latent shape: {self.current_latent.shape}')
		# print(f'Current prop shape: {self.current_prop.shape}')
		return (
			representation, # based on world model
			reward, # from world model
			terminated, # For now only based on step count
			truncated, # Truncated
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

def generate_data(vq, lstm, n_sample=1000, policy=None, training_set=True):
	base_images_path = 'data/pusher/imgs' + ('_tr/' if training_set else '_vl/')
	action_path = 'data/pusher/action_reward_data' + ('_tr.json' if training_set else '_vl.json')
	actions = []
	rewards = []
	proprioception = []
	if os.path.exists(action_path):
		with open(action_path, "r") as f:
			f = json.load(f)
			actions = f['actions']
			rewards = f['reward']
			proprioception = f['proprioception']
	if not os.path.exists(base_images_path):
		os.makedirs(base_images_path)
		
	env = PusherWrapEnv(vq, lstm)
	obs, _ = env.reset()
	step = 0
	episode = len(actions)
	print(episode)
	actions.append([])
	rewards.append([])
	proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_images_path + f'img_{episode}_{step}.png')
	for i in tqdm(range(n_sample)):
		step += 1
		if policy == None:
			action = env.action_space.sample()
		else:
			action, _ = policy.predict(obs, deterministic=False)
		obs, rew, ter, trunc, _ = env.step(action)
		proprioception[-1].append(env.current_prop.flatten().tolist())
		actions[-1].append(action.tolist())
		env.current_render.save(base_images_path + f'img_{episode}_{step}.png')
		rewards[-1].append(float(rew))
		if ter or trunc:
			obs, info = env.reset()
			if i < n_sample - 1:
				episode += 1
				step = 0
				proprioception.append([env.current_prop.flatten().tolist()])
				env.current_render.save(base_images_path + f'img_{episode}_{step}.png')
				actions.append([])
				rewards.append([])
	with open(action_path, "w") as f:
		json.dump(
			{
				"actions": actions,
				"reward": rewards,
				"proprioception": proprioception
			},
			f,
			indent=4
		)

if __name__ == "__main__":
	SMOOTH = False
	KL = False
	vq = load_vq_vae(PUSHER, 64, 16, 4, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), 1024, SMOOTH, True, KL)
	env = PusherWrapEnv(vq, lstm)
	observation, _ = env.reset()
	env.render()
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(PUSHER['models'] + 'agent', env)
	while not done:
		#action = env.action_space.sample()  # random action
		action, _states = agent.predict(observation, deterministic=False)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		env.render()
		done = terminated or truncated
		total_reward += reward
		step_count += 1
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()