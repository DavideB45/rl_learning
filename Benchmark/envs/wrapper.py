import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import metaworld
import torch
import torchvision.transforms as T
import json
from PIL import Image
from stable_baselines3.ppo import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from helpers.data import get_data_path
from global_var import *

class MetaWrapEnv(gym.Env):
	"""
	This environemt is a wrapper of the real environment used at inference time
	Since the agent can't do inference directly on the data coming from the environment
	"""


	def __init__(self, vq:VQVAE=None, lstm:LSTMQuantized=None):
		super(MetaWrapEnv, self).__init__()

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.lstm = lstm
		self.lstm.eval()

		self.env = gym.make('Meta-World/MT1', env_name=CURRENT_ENV['env_name'],
				render_mode='rgb_array', camera_id=CURRENT_ENV['camera_id'], width = 128, height = 128)
		self.mu = vq.quantizer.embedding.weight.data.mean()
		self.std = vq.quantizer.embedding.weight.data.std()
		self.action_space = spaces.Box(
			low=-1, high=1, 
			shape=(4,), 
			dtype=np.float32
		)
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
		img = self.env.render()
		img = Image.fromarray(img)
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
		self.current_prop = torch.tensor(prop[:4], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		self.current_render = img
		normalized = (self.current_latent.flatten()-self.mu)/self.std
		representation = torch.cat([normalized, self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		return representation, {}

	def step(self, action,) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		prop, reward, terminated, truncated, info = self.env.step(action)
		prop = prop[0:4]
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
		normalized = (self.current_latent.flatten()-self.mu)/self.std
		representation = torch.cat([normalized, self.hidden_state[0].flatten()], dim=-1).cpu().numpy()
		return (
			representation, # based on world model
			reward, # from world model
			terminated, # For now only based on step count
			truncated, # Truncated
			info # empty dict
		)
	
	def render(self):
		if True:
			with torch.no_grad():
				img = self.vq.decode(self.current_latent).squeeze(0).permute(1, 2, 0).cpu().numpy()
				img = (img * 255).astype(np.uint8)
				image = Image.fromarray(img)
				image = self.get_img() 
				image_resized = image.resize((512, 512), Image.NEAREST)
				cv2.imshow('DreamEnv', np.array(image_resized))
				cv2.waitKey(100)
				return image_resized
		else:
			return self.current_render
		
	def close(self):
		self.env.close()
		pass

def generate_data(vq:VQVAE, lstm:LSTMQuantized, n_sample:int=1000, policy:BaseAlgorithm=None, training_set:bool=True, round:int=0):
	base_path = get_data_path(CURRENT_ENV['img_dir'], training_set, round)
	action_path = base_path + TRANSITIONS
	actions = []
	rewards = []
	proprioception = []
	if os.path.exists(action_path):
		with open(action_path, "r") as f:
			f = json.load(f)
			actions = f['actions']
			rewards = f['reward']
			proprioception = f['proprioception']
	if not os.path.exists(base_path):
		os.makedirs(base_path)
	if not os.path.exists(CURRENT_ENV['models']):
		os.makedirs(CURRENT_ENV['models'])
		
	env = MetaWrapEnv(vq, lstm)
	obs, _ = env.reset(seed=0)
	step = 0
	episode = len(actions)
	print(episode)
	actions.append([])
	rewards.append([])
	proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')
	#for i in tqdm(range(n_sample)):
	for i in range(n_sample):
		step += 1
		if policy == None:
			action = env.action_space.sample()
		else:
			# qui c'è un problema quando si usa gSDE
			action, _ = policy.predict(obs, deterministic=False)
		obs, rew, ter, trunc, _ = env.step(action)
		proprioception[-1].append(env.current_prop.flatten().tolist())
		actions[-1].append(action.tolist())
		env.current_render.save(base_path + f'img_{episode}_{step}.png')
		rewards[-1].append(float(rew))
		if ter or trunc:
			obs, info = env.reset()
			if i < n_sample - 1:
				episode += 1
				step = 0
				proprioception.append([env.current_prop.flatten().tolist()])
				env.current_render.save(base_path + f'img_{episode}_{step}.png')
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

def evaluate_gathering(vq:VQVAE, lstm:LSTMQuantized, policy:BaseAlgorithm, n_sample:int=1000, training_set:bool=True, round:int=0) -> tuple[list[float], list[bool]]:
	"""
	Evaluate the policy on the environment, gathering data and saving it in the same format as generate_data
	Args:
		vq: VQVAE model
		lstm: LSTM model
		n_sample: number of samples to gather
		policy: policy to use for action selection, if None random actions will be taken
		training_set: whether to use the training set or the test set path for data storage
		round: round number for data storage (only zero should be used at the current moment and possibly forever)
	Returns:
		tuple[list[float], list[bool]]: total rewards and success flags for each episode
	"""
	base_path = get_data_path(CURRENT_ENV['img_dir'], training_set, round)
	action_path = base_path + TRANSITIONS
	actions = []
	rewards = []
	proprioception = []
	if os.path.exists(action_path):
		with open(action_path, "r") as f:
			f = json.load(f)
			actions = f['actions']
			rewards = f['reward']
			proprioception = f['proprioception']
	if not os.path.exists(base_path):
		os.makedirs(base_path)
	if not os.path.exists(CURRENT_ENV['models']):
		os.makedirs(CURRENT_ENV['models'])
		
	env = MetaWrapEnv(vq, lstm)
	obs, _ = env.reset()
	step = 0
	episode = len(actions)
	print("Number of episodes in history:", episode)
	actions.append([]), rewards.append([]), proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')
	tot_rewards = [0]
	tot_success = [False]
	for i in range(n_sample):
		step += 1
		if step % 10 == 0: # SB3 does not do this automatically since we are evaluating the model
			policy.policy.reset_noise()
		action, _ = policy.predict(obs, deterministic=False)
		obs, rew, ter, trunc, info = env.step(action)
		proprioception[-1].append(env.current_prop.flatten().tolist()), actions[-1].append(action.tolist()), rewards[-1].append(float(rew))
		env.current_render.save(base_path + f'img_{episode}_{step}.png')
		tot_rewards[-1] += rew
		tot_success[-1] = (info['success'] == 1) or tot_success[-1]
		if ter or trunc:
			obs, info = env.reset()
			if i < n_sample - 1:
				episode += 1
				step = 0
				proprioception.append([env.current_prop.flatten().tolist()]), actions.append([]), rewards.append([]), tot_rewards.append(0), tot_success.append(False)
				env.current_render.save(base_path + f'img_{episode}_{step}.png')
	with open(action_path, "w") as f:
		json.dump(
			{ "actions": actions, "reward": rewards, "proprioception": proprioception },
			f, indent=4
		)
	return tot_rewards, tot_success

if __name__ == "__main__":
	from random import randint
	SMOOTH = True if SMOOTH > 0 else False
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTH, False, False)
	env = MetaWrapEnv(vq, lstm)
	observation, _ = env.reset()
	frames = []
	frames.append(env.render().rotate(180))
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(CURRENT_ENV['models'] + 'agent', env)
	while not done:
		if randint(0, 9) < -1:
			action = env.action_space.sample()  # random action
		else:
			action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward}")
		frames.append(env.render().rotate(180))
		done = terminated or truncated
		total_reward += reward
		step_count += 1
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()

	GIF_PATH = "output.gif"
	FRAME_DURATION_MS = 2
	frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        loop=0,                    # 0 = loop forever
        duration=FRAME_DURATION_MS,
    )