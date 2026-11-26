import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torchvision.transforms as T
import json
from PIL import Image
from stable_baselines3.ppo import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from gym_tasks.grasp_env import GraspEnv  # eleanor-sim, installed with `pip install -e "<eleanor-sim>[gym]"`

from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from helpers.data import get_data_path
from global_var import *

class GraspWrapEnv(gym.Env):
	"""
	Wrapper of the real environment (the ELEANOR grasp task of eleanor-sim) used at inference time,
	since the agent can't act directly on the data coming from the environment.

	Observation: vq latent of the rendered frame (normalized) + lstm hidden state + proprioception.
	Proprioception is the 7 tendon actuations only: the object position, which GraspEnv appends to its
	observation, is privileged and is dropped.

	A simulation that diverges (GraspEnv raises FloatingPointError) truncates the episode with
	info['diverged'] = True; the gathering functions below discard such episodes.
	"""

	def __init__(self, vq:VQVAE=None, dyn:LSTMQuantized=None):
		super(GraspWrapEnv, self).__init__()

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.dyn = dyn
		self.dyn.eval()

		self.env = GraspEnv(render_mode='rgb_array', camera=CURRENT_ENV['camera'],
				yaw_range=CURRENT_ENV['yaw_range'], randomize_defects=CURRENT_ENV['randomize_defects'],
				image_size=CURRENT_ENV['render_size'])
		self.ep_len = GraspEnv.T
		self.mu = vq.quantizer.embedding.weight.data.mean()
		self.std = vq.quantizer.embedding.weight.data.std()
		self.action_space = spaces.Box(
			low=-1, high=1,
			shape=(CURRENT_ENV['a_size'],),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.vq_dim + self.dyn.hidden_dim + self.dyn.prop_dim,), dtype=np.float32
		)
		self.to_tensor_ = T.ToTensor()

	def get_img(self) -> Image.Image:
		'''
		Renders the current frame of the environment (already at CURRENT_ENV['render_size'])
		'''
		img = self.env.render()
		if not np.isfinite(img).all():
			print("BAD IMAGE")
			raise RuntimeError()
		return Image.fromarray(img)

	def _encode(self, img:Image.Image) -> torch.Tensor:
		t_img = self.to_tensor_(img).unsqueeze(0).to(self.vq.device)
		_, lat, _ = self.vq.quantize(self.vq.encode(t_img))
		return lat

	def _representation(self) -> np.ndarray:
		latent = (self.current_latent.flatten()-self.mu)/self.std
		representation = torch.cat([latent.cpu(), self.hidden_state[0].cpu().flatten(), self.current_prop.flatten()], dim=-1)
		if not torch.isfinite(representation).all():
			print("OBS NAN")
			raise RuntimeError()
		return representation.numpy()

	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		returns: initial observation obtained encoding the first image, a zero hidden state and the initial actuations;
		info carries the 'yaw' of the start pose
		'''
		super().reset(seed=seed, options=options)
		obs, info = self.env.reset(seed=seed, options=options)
		img = self.get_img()
		with torch.no_grad():
			self.current_latent = self._encode(img)
			self.hidden_state = (torch.zeros(1, 1, self.dyn.hidden_dim).to(self.vq.device),
				torch.zeros(1, 1, self.dyn.hidden_dim).to(self.vq.device))
		self.current_render = img
		self.current_prop = torch.tensor(obs[:PROP_SIZE], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		self.last_obs = self._representation()
		return self.last_obs, info

	def step(self, action) -> tuple:
		'''
		Step in the real environment and advance the LSTM hidden state by one step
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		if not np.isfinite(action).all():
			print("BAD ACTION")
			print(action)
			raise RuntimeError()
		try:
			obs, reward, terminated, truncated, info = self.env.step(action)
		except FloatingPointError:
			print("[WARNING] the simulation diverged, truncating the episode")
			return self.last_obs, 0.0, False, True, {'diverged': True, 'success': 0.0}
		img = self.get_img()
		with torch.no_grad():
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			_, _, _, _, h = self.dyn.forward(self.current_latent.unsqueeze(0).to(self.vq.device), action_tensor.to(self.vq.device), self.current_prop.to(self.vq.device), self.hidden_state)
			self.hidden_state = h
			self.current_latent = self._encode(img)
		self.current_render = img
		self.current_prop = torch.tensor(obs[:PROP_SIZE], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		self.last_obs = self._representation()
		return self.last_obs, float(reward), terminated, truncated, info

	def render(self):
		return self.current_render.resize((512, 512), Image.NEAREST)

	def decoded_render(self):
		'''what the VQ-VAE makes of the current frame'''
		with torch.no_grad():
			img = self.vq.decode(self.current_latent).squeeze(0).permute(1, 2, 0).cpu().numpy()
		return Image.fromarray((img * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)

	def close(self):
		self.env.close()


def _gather(vq:VQVAE, lstm:LSTMQuantized, policy:BaseAlgorithm, n_sample:int, training_set:bool, round:int) -> tuple[list[float], list[bool]]:
	'''
	Roll out `policy` (random actions if None) for whole episodes, at least n_sample steps in total, and append
	them to the dataset in data/<env>/: images img_{episode}_{step}.png and, in action_reward_data.json, per episode
	ep_len actions, ep_len rewards and ep_len+1 proprioception vectors. Diverged episodes are discarded and redone
	(their images are overwritten by the next episode).
	Returns the return and the success of each kept episode.
	'''
	base_path = get_data_path(CURRENT_ENV['img_dir'], training_set, round)
	action_path = base_path + TRANSITIONS
	actions, rewards, proprioception = [], [], []
	if os.path.exists(action_path):
		with open(action_path, "r") as f:
			f = json.load(f)
			actions = f['actions']
			rewards = f['reward']
			proprioception = f['proprioception']
	os.makedirs(base_path, exist_ok=True)
	os.makedirs(CURRENT_ENV['models'], exist_ok=True)

	env = GraspWrapEnv(vq, lstm)
	n_episodes = -(-n_sample // env.ep_len)  # ceil: always whole episodes
	print("Number of episodes in history:", len(actions), f"| gathering {n_episodes} more")
	tot_rewards, tot_success = [], []
	n_diverged = 0
	while len(tot_rewards) < n_episodes:
		episode = len(actions)
		obs, _ = env.reset()
		ep_act, ep_rew, ep_prop = [], [], [env.current_prop.flatten().tolist()]
		env.current_render.save(base_path + f'img_{episode}_0.png')
		done, diverged, success, step = False, False, False, 0
		while not done:
			step += 1
			if policy is None:
				action = env.action_space.sample()
			else:
				if step % 10 == 0: # SB3 does not do this automatically since we are evaluating the model
					policy.policy.reset_noise()
				action, _ = policy.predict(obs, deterministic=False)
			obs, rew, ter, trunc, info = env.step(action)
			if info.get('diverged', False):
				diverged = True
				break
			ep_act.append(action.tolist()), ep_rew.append(float(rew)), ep_prop.append(env.current_prop.flatten().tolist())
			env.current_render.save(base_path + f'img_{episode}_{step}.png')
			success = success or info['success'] == 1
			done = ter or trunc
		if diverged:
			n_diverged += 1
			continue
		actions.append(ep_act), rewards.append(ep_rew), proprioception.append(ep_prop)
		tot_rewards.append(sum(ep_rew)), tot_success.append(success)
	if n_diverged:
		print(f"[WARNING] {n_diverged} diverged episodes discarded")
	env.close()
	with open(action_path, "w") as f:
		json.dump(
			{ "actions": actions, "reward": rewards, "proprioception": proprioception },
			f, indent=4
		)
	return tot_rewards, tot_success

def generate_data(vq:VQVAE, lstm:LSTMQuantized, n_sample:int=1000, policy:BaseAlgorithm=None, training_set:bool=True, round:int=0):
	'''Gather at least n_sample steps (whole episodes) with `policy`, or random actions if None'''
	_gather(vq, lstm, policy, n_sample, training_set, round)

def evaluate_gathering(vq:VQVAE, lstm:LSTMQuantized, policy:BaseAlgorithm, n_sample:int=1000, training_set:bool=True, round:int=0) -> tuple[list[float], list[bool]]:
	"""
	Evaluate the policy on the environment, gathering data and saving it in the same format as generate_data
	Returns:
		tuple[list[float], list[bool]]: total rewards and success flags for each episode
	"""
	return _gather(vq, lstm, policy, n_sample, training_set, round)

if __name__ == "__main__":
	if 'MUJOCO_GL' not in os.environ and sys.platform != 'darwin':
		os.environ['MUJOCO_GL'] = 'egl'
	SMOOTH = True if SMOOTH > 0 else False
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTH, False, False)
	env = GraspWrapEnv(vq, lstm)
	observation, info = env.reset()
	print(f"yaw {info['yaw']:+.1f} deg")
	frames = [env.render()]
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(CURRENT_ENV['models'] + 'agent' + f'{EXP_ID}', env)
	while not done:
		action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward:+.3f} | action: {action}")
		frames.append(env.render())
		done = terminated or truncated
		total_reward += reward
		step_count += 1
	print(f"Total reward: {total_reward:+.2f}, success: {info['success'] == 1}")
	env.close()

	GIF_PATH = "output.gif"
	frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], loop=0, duration=200)
