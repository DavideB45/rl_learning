import cv2
import numpy as np
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
from dynamics.transformer import TransformerArc
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import *

class MetaDreamEnv(VecEnv):
	"""
	Completely simulated environment using the VAE and MDRNN models
	The starting state is obtained from the real environment
	Then the environment is simulated using only the world model
	"""


	def __init__(self, vq:VQVAE, dynamic:LSTMQClass | TransformerArc, dataloader:DataLoader, init_len:int=1, ep_len:int=20, num_envs: int = 1):
		
		self.num_envs = num_envs
		self.max_len = ep_len # this way the model will learn only 20 steps, hopefully in the end he will manage to merge his knowledge
		self.step_count = 0
		self.i_len = init_len

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.dyn = dynamic
		self.using_tr = isinstance(dynamic, TransformerArc)
		self.dyn.eval()
		self.hidden_state = None # (num_envs, hidden_dim)
		self.mu = vq.quantizer.embedding.weight.data.mean()
		self.std = vq.quantizer.embedding.weight.data.std()

		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, 
			shape=(self.vq_dim + (self.dyn.hidden_dim if self.using_tr else 0),), 
			dtype=np.float32
		)
		self.action_space = spaces.Box(
			low=-1, high=1, 
			shape=(4,),
			dtype=np.float32
		)
		super(MetaDreamEnv, self).__init__(
			num_envs=num_envs,
			action_space=self.action_space,
			observation_space=self.observation_space	
		)

		self.data = dataloader

	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		seed: random seed
		options: additional options
		returns: initial observation (np.array) obtained encoding the first image and the initial hidden state
		'''
		if seed is not None:
			print("[WARNING] I haven't implemented seed it's always random")
		init_data_list = [self.data.dataset[np.random.randint(len(self.data.dataset))] for _ in range(self.num_envs)]
		with torch.no_grad():
			latents = torch.stack([init_data['latent'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)
			actions = torch.stack([init_data['action'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)
			props = torch.stack([init_data['proprioception'][:self.i_len, :] for init_data in init_data_list]).to(self.vq.device)

			if self.using_tr:
				_, pred, _, _ = self.dyn.forward(latents, actions) # wrong because the length has a cap
				self.current_latent = pred[:, -1, :, :, :]
				latent_flat = (self.current_latent.reshape(self.num_envs, -1)-self.mu)/self.std # see comments below
				representation = latent_flat.cpu().numpy()
			else:
				_, pred, prop, _, h = self.dyn.forward(latents, actions, props, None)
				self.hidden_state = h
				hidden_flat = self.hidden_state[0].reshape(self.num_envs, -1)
				self.current_latent = pred[:, -1, :, :, :]
				latent_flat = (self.current_latent.reshape(self.num_envs, -1)-self.mu)/self.std
				self.current_prop = prop[:, -1, :]
				representation = torch.cat([latent_flat, hidden_flat], dim=-1).cpu().numpy()
			
		self.step_count = 0
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
			
			if self.using_tr:
				action_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(self.vq.device) # change this to use old actions
				latent_input = self.current_latent.unsqueeze(1).to(self.vq.device) # change this to a list of tensors
				_, pred, _, _ = self.dyn.forward(latent_input, actions) # needs to be updatet because we are taking only 1 state now split in if else
				latent_flat = (self.current_latent.reshape(self.num_envs, -1)-self.mu)/self.std
				self.current_latent = pred[:, -1, :, :, :]# chage this keeping only crrect number of stuff
				#also add action history
				representation = latent_flat.cpu().numpy()
			else:
				action_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(self.vq.device)
				latent_input = self.current_latent.unsqueeze(1).to(self.vq.device)
				prop_input = self.current_prop.unsqueeze(1).to(self.vq.device)
				_, pred, prop, rew, h = self.dyn.forward(latent_input, action_tensor, prop_input, self.hidden_state)
				self.hidden_state = h
				hidden_flat = self.hidden_state[0].reshape(self.num_envs, -1)
				latent_flat = (self.current_latent.reshape(self.num_envs, -1)-self.mu)/self.std
				self.current_latent = pred[:, -1, :, :, :]
				self.current_prop = prop[:, -1, :]
				representation = torch.cat([latent_flat, hidden_flat], dim=-1).cpu().numpy()

			self.step_count += 1
			terminateds = np.array([self.step_count >= self.max_len] * self.num_envs, dtype=bool)
			infos = [
				{'terminal_observation': representation[i]} for i in range(self.num_envs)
			]
		return (
			representation if not terminateds.any() else self.reset(), # based on world model
			np.array(rew.flatten().cpu()), # from world model
			terminateds, # For now only based on step count
			infos
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
		pass

	# additional implementation for the interface
	def env_is_wrapped(self, wrapper_class, indices = None):
		return super().env_is_wrapped(wrapper_class, indices)
	
	def env_method(self, method_name, *method_args, indices = None, **method_kwargs):
		return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
	
	def get_attr(self, attr_name, indices = None):
		if 'render_mode':
			return ['rgb_array' for _ in range(self.num_envs)]
		return super().get_attr(attr_name, indices)
	
	def set_attr(self, attr_name, value, indices = None):
		return super().set_attr(attr_name, value, indices)
	
	def step_async(self, actions):
		# this is a trick: the call is not really async so this can cause some slowdown
		self.async_values = self.step(actions)
	
	def step_wait(self):
		return self.async_values
	


if __name__ == "__main__":
	SMOOTH = True if SMOOTH > 0 else False
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTH, False, False)
	env = MetaDreamEnv(vq=vq, lstm=lstm, dataloader=make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, 0), vq, 100, 1), 
					  num_envs=1, ep_len=50, init_len=1)
	observation = env.reset()
	frames = []
	frames.append(env.render().rotate(180))
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(CURRENT_ENV['models'] + 'agent', env)
	while not done:
		#action = env.action_space.sample()  # random action
		action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		frames.append(env.render().rotate(180))
		done = terminated.any()
		total_reward += reward
		step_count += 1
		#time.sleep(2)
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()

	GIF_PATH = "output.gif"
	FRAME_DURATION_MS = 50
	frames[0].save(
		GIF_PATH,
		save_all=True,
		append_images=frames[1:],
		loop=0,                    # 0 = loop forever
		duration=FRAME_DURATION_MS,
	)