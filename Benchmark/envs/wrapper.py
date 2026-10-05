import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torchvision.transforms as T
import json
from PIL import Image
from stable_baselines3.ppo import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from envs.physical.realWorld import RealWorld
from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from helpers.data import get_data_path
from global_var import *

class SoftWrapEnv(gym.Env):
	"""
	This environemt is a wrapper of the real environment used at inference time
	Since the agent can't do inference directly on the data coming from the environment
	"""


	def __init__(self, vq:VQVAE=None, dyn:LSTMQuantized=None):
		'''
		initialization of the wrapper to use the models trained in MetaDreamEnv

		if lstm is None only the vq latent representation will be used in the representaiotn 
		this is useful if the dynamic model is a transformer based model and does not have a
		latent space that represents the past
		'''
		super(SoftWrapEnv, self).__init__()

		self.vq = vq
		self.vq.eval()
		self.vq_dim = self.vq.latent_dim**2*self.vq.code_depth
		self.dyn = dyn
		self.dyn.eval()
		self.env = RealWorld(view_camera_id=1, width = 480, height = 480, cropped_width=64, cropped_height=64, camera_hz=20,
				aruco_camera_id=0, marker_id=5, min_sharpness=100.0, aruco_hz=20,
				render_mode='human', max_steps=150, env_hz=10, debug=True, rew_multiplier=30.0)
		self.mu = vq.quantizer.embedding.weight.data.mean()
		self.std = vq.quantizer.embedding.weight.data.std()
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.vq_dim + self.dyn.hidden_dim + self.dyn.prop_dim,), dtype=np.float32
		)
		self.to_tensor_ = T.ToTensor()

	def save_next_episode(self, id):
		self.env.save_next_episode_video(id)

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
		if not np.isfinite(img).all():
			print("BAD IMAGE")
			raise RuntimeError()
		img = Image.fromarray(img)
		img = img.resize((CURRENT_ENV["render_size"], CURRENT_ENV["render_size"]))
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
			h = (torch.zeros(1, 1, self.dyn.hidden_dim).to(self.vq.device),
				torch.zeros(1, 1, self.dyn.hidden_dim).to(self.vq.device))
			self.hidden_state = h
		self.current_render = img
		
		self.current_latent = lat
		representation = (self.current_latent.flatten()-self.mu)/self.std
		
		self.current_prop = torch.tensor(prop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

		representation = torch.cat([representation.cpu(), self.hidden_state[0].cpu().flatten(), self.current_prop.flatten()], dim=-1)
		return representation.cpu().numpy(), {}

	def step(self, action,) -> tuple:
		'''
		Step in the environment using only MDRNN
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		if not np.isfinite(action).all():
			print("BAD ACTION")
			print(action)
			raise RuntimeError()
		if np.max(np.abs(action)) > 20:
			print("STRANGE ACTION")
			print(action)
		prop, reward, terminated, truncated, info = self.env.step(action)
		img = self.get_img()
		with torch.no_grad():
			t_img = self.to_tensor_(img).unsqueeze(0).to(self.vq.device)
			_, lat, _ = self.vq.quantize(self.vq.encode(t_img))
			
			action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			_, _, _, _, h = self.dyn.forward(self.current_latent.unsqueeze(0).to(self.vq.device), action_tensor.to(self.vq.device), self.current_prop.to(self.vq.device), self.hidden_state)
			self.hidden_state = h
			self.current_latent = lat
			representation = (self.current_latent.flatten()-self.mu)/self.std
			
			self.current_render = img
			self.current_prop = torch.tensor(prop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			representation = torch.cat([representation.cpu(), self.hidden_state[0].cpu().flatten(), self.current_prop.flatten()], dim=-1)
			if not torch.isfinite(representation).all():
				print("OBS NAN")
				raise RuntimeError()
		return (
			representation.cpu().numpy(), # based on world model
			reward, # from world model
			terminated, # For now only based on step count
			truncated, # Truncated
			info # empty dict
		)
	
	def render(self):
		if False:
			with torch.no_grad():
				img = self.vq.decode(self.current_latent[:, :, :]).squeeze(0).permute(1, 2, 0).cpu().numpy()
				img = (img * 255).astype(np.uint8)
				image = Image.fromarray(img)
				image = self.get_img() 
				image_resized = image.resize((512, 512), Image.NEAREST)
				#cv2.imshow('DreamEnv', np.array(image_resized))
				#cv2.waitKey(100)
				return image_resized
		else:
			return self.env.render()
		
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
		
	env = SoftWrapEnv(vq, lstm)
	obs, _ = env.reset()
	step = 0
	episode = len(actions)
	print(episode)
	actions.append([])
	rewards.append([])
	proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')

	# goal oriented blabbing: when there is no policy yet, draw lines through pressure
	# space instead of sampling i.i.d. random actions each step, so this first round of
	# data actually sweeps the workspace instead of jittering in place (same idea as in
	# envs/physical/realWorld.py's __main__)
	rng = np.random.default_rng()
	LINE_NOISE_STD = 0.15          # exploration noise added on top of the line direction
	LINE_LEN_RANGE = (5, 15)       # steps a line segment lasts before a new target is picked

	def new_line_target():
		return rng.uniform(0.0, env.env.max_pressure, size=3).astype(np.float32)

	line_target = new_line_target()
	line_steps_left = rng.integers(*LINE_LEN_RANGE)

	for i in range(n_sample):
		step += 1
		if policy == None:
			if line_steps_left <= 0:
				line_target = new_line_target()
				line_steps_left = rng.integers(*LINE_LEN_RANGE)
			line_steps_left -= 1
			direction = (line_target - env.env.current_pressure) / 0.1
			noise = rng.normal(0.0, LINE_NOISE_STD, size=3)
			action = np.clip(direction + noise, -1.0, 1.0).astype(np.float32)
		else:
			if step % 10 == 0: # SB3 does not do this automatically since we are evaluating the model
				policy.policy.reset_noise()
			action, _ = policy.predict(obs, deterministic=False)
		obs, rew, ter, trunc, _ = env.step(action)
		proprioception[-1].append(env.current_prop.flatten().tolist())
		actions[-1].append(action.tolist())
		env.current_render.save(base_path + f'img_{episode}_{step}.png')
		rewards[-1].append(float(rew))
		if ter or trunc:
			obs, info = env.reset()
			line_target = new_line_target()
			line_steps_left = rng.integers(*LINE_LEN_RANGE)
			time.sleep(2)
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

class VirtualJoystick:
	'''
	Circular on-screen joystick (OpenCV window) used by generate_data_interactive.
	Click and drag the knob with the mouse/trackpad, on release it snaps back to the center (zero pressure).
	The three chambers sit 120 deg apart (C1 upper left, C2 upper right, C3 bottom) and the angle picks
	the mix: a chamber is at max within 60 deg of its direction (so halfway between two chambers both
	are at max) and fades linearly to zero at the neighbouring chambers. The distance from the center
	scales the pressure linearly.
	'''
	WINDOW = "Joystick"
	CHAMBER_ANGLES = np.array([150.0, 30.0, 270.0]) # degrees, counter-clockwise from the right (screen view)
	CHAMBER_COLORS = [(60, 180, 255), (80, 220, 100), (255, 140, 80)] # BGR
	TEXT_HEIGHT = 50 # space under the disk for the pressure readout

	def __init__(self, max_pressure:float, radius:int=180, margin:int=40):
		self.max_pressure = max_pressure
		self.radius = radius
		self.margin = margin
		self.width = 2 * (radius + margin)
		self.height = 2 * (radius + margin) + self.TEXT_HEIGHT
		self.center = (radius + margin, radius + margin)
		self.knob = self.center
		self.dragging = False
		cv2.namedWindow(self.WINDOW)
		cv2.setMouseCallback(self.WINDOW, self._on_mouse)
		cv2.moveWindow(self.WINDOW, 230, 530) # next to RealWorld's "Pressure" window
		self.draw()

	def _clamp(self, x, y):
		'''keeps the knob inside the disk'''
		dx, dy = x - self.center[0], y - self.center[1]
		r = np.hypot(dx, dy)
		if r > self.radius:
			dx, dy = dx * self.radius / r, dy * self.radius / r
		return (int(round(self.center[0] + dx)), int(round(self.center[1] + dy)))

	def _on_mouse(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.dragging = True
			self.knob = self._clamp(x, y)
		elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
			self.knob = self._clamp(x, y)
		elif event == cv2.EVENT_LBUTTONUP:
			self.dragging = False
			self.knob = self.center

	def _on_screen(self, angle_deg, r):
		'''pixel position at angle_deg (counter-clockwise from the right, as seen on screen) and distance r'''
		a = np.radians(angle_deg)
		return (int(round(self.center[0] + r * np.cos(a))), int(round(self.center[1] - r * np.sin(a))))

	def target_pressure(self) -> np.ndarray:
		'''
		Returns:
			np.ndarray: target pressure for the 3 chambers selected by the current knob position
		'''
		dx = self.knob[0] - self.center[0]
		dy = self.center[1] - self.knob[1]
		r = min(np.hypot(dx, dy) / self.radius, 1.0)
		angle = np.degrees(np.arctan2(dy, dx))
		diff = np.abs((angle - self.CHAMBER_ANGLES + 180.0) % 360.0 - 180.0) # angular distance to each chamber
		weights = np.clip((120.0 - diff) / 60.0, 0.0, 1.0)
		return weights.astype(np.float32) * r * self.max_pressure

	def draw(self, current_pressure=None):
		'''redraws the joystick window and pumps the OpenCV event loop (so the mouse callback runs)'''
		img = np.full((self.height, self.width, 3), 30, dtype=np.uint8)
		cv2.circle(img, self.center, self.radius, (55, 55, 55), -1, cv2.LINE_AA)
		cv2.circle(img, self.center, self.radius, (140, 140, 140), 2, cv2.LINE_AA)
		cv2.circle(img, self.center, self.radius // 2, (90, 90, 90), 1, cv2.LINE_AA)
		for i, (a, color) in enumerate(zip(self.CHAMBER_ANGLES, self.CHAMBER_COLORS)):
			cv2.line(img, self.center, self._on_screen(a, self.radius), (100, 100, 100), 1, cv2.LINE_AA)
			cv2.circle(img, self._on_screen(a, self.radius), 5, color, -1, cv2.LINE_AA)
			x, y = self._on_screen(a, self.radius + 22)
			cv2.putText(img, f"C{i + 1}", (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		cv2.line(img, self.center, self.knob, (200, 200, 200), 2, cv2.LINE_AA)
		cv2.circle(img, self.knob, 16, (0, 200, 255) if self.dragging else (180, 180, 180), -1, cv2.LINE_AA)
		target = self.target_pressure()
		lines = ["target " + "  ".join(f"C{i + 1} {p:.2f}" for i, p in enumerate(target))]
		if current_pressure is not None:
			lines.append("now    " + "  ".join(f"C{i + 1} {p:.2f}" for i, p in enumerate(current_pressure[:3])))
		for j, text in enumerate(lines):
			cv2.putText(img, text, (10, self.height - self.TEXT_HEIGHT + 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imshow(self.WINDOW, img)
		cv2.waitKey(1)

	def wait(self, seconds:float, current_pressure=None):
		'''like time.sleep but keeps the joystick responsive'''
		end = time.time() + seconds
		while time.time() < end:
			self.draw(current_pressure)
			time.sleep(0.02)

	def close(self):
		cv2.destroyWindow(self.WINDOW)

def generate_data_interactive(vq:VQVAE, lstm:LSTMQuantized, n_sample:int=1000, training_set:bool=True, round:int=0):
	'''
	Same as generate_data (same episodes, same storage format) but the actions come from a human
	driving the VirtualJoystick. The joystick sets a target pressure, the stored action is the
	pressure delta (clipped to [-1, 1]) that moves the robot towards it, as RealWorld.step expects.
	Args:
		vq: VQVAE model
		lstm: LSTM model
		n_sample: number of steps to gather
		training_set: whether to use the training set or the test set path for data storage
		round: round number for data storage
	'''
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

	env = SoftWrapEnv(vq, lstm)
	joystick = VirtualJoystick(env.env.max_pressure)
	obs, _ = env.reset()
	step = 0
	episode = len(actions)
	print(episode)
	actions.append([])
	rewards.append([])
	total_reward = 0
	proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')

	for i in tqdm(range(n_sample)):
		step += 1
		joystick.draw(env.env.current_pressure)
		target = joystick.target_pressure()
		# 0.1 is the pressure gain per unit of action in RealWorld.step
		action = np.clip((target - env.env.current_pressure) / 0.1, -1.0, 1.0).astype(np.float32)
		obs, rew, ter, trunc, _ = env.step(action)
		proprioception[-1].append(env.current_prop.flatten().tolist())
		actions[-1].append(action.tolist())
		env.current_render.save(base_path + f'img_{episode}_{step}.png')
		rewards[-1].append(float(rew))
		total_reward += rew
		if ter or trunc:
			obs, info = env.reset()
			joystick.wait(2, env.env.current_pressure)
			if i < n_sample - 1:
				episode += 1
				step = 0
				proprioception.append([env.current_prop.flatten().tolist()])
				env.current_render.save(base_path + f'img_{episode}_{step}.png')
				actions.append([])
				rewards.append([])
	joystick.close()
	env.close()
	print(f'Total reward = {total_reward}')
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
		
	env = SoftWrapEnv(vq, lstm)
	obs, _ = env.reset()
	step = 0
	episode = len(actions)
	print("Number of episodes in history:", episode)
	actions.append([]), rewards.append([]), proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')
	tot_rewards = [0]
	tot_success = [False]
	for i in tqdm(range(n_sample)):
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

def evaluate_gathering_safe(vq, lstm, policy, n_sample:int=1000, training_set:bool=True, round:int=0, save_id:str|None|int=None, auto_accept:bool=True) -> tuple[list[float], list[bool]]:
	"""
	Evaluate the policy on the environment, gathering data and saving it in the same format as generate_data
	Args:
		vq: VQVAE model
		lstm: LSTM model
		n_sample: number of valid samples to gather
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
		
	env = SoftWrapEnv(vq, lstm)
	obs, _ = env.reset()
	if save_id is not None:
		# must be armed *after* the initial reset: reset() itself calls save_now(),
		# which would otherwise immediately consume/clear the flag before any frame
		# of the episode is captured
		env.save_next_episode(save_id)
	step = 0
	episode = len(actions)
	print("Number of episodes in history:", episode)
	
	actions.append([])
	rewards.append([])
	proprioception.append([env.current_prop.flatten().tolist()])
	env.current_render.save(base_path + f'img_{episode}_{step}.png')
	
	tot_rewards = [0]
	tot_success = [False]
	
	# Use a while loop so we can rollback the counter if an episode is discarded
	i = 0
	with tqdm(total=n_sample) as pbar:
		while i < n_sample:
			step += 1
			i += 1
			pbar.update()
			
			if step % 10 == 0 and policy is not None: 
				policy.policy.reset_noise()

			if policy is not None:
				action, _ = policy.predict(obs, deterministic=False)
			else:
				action = env.action_space.sample()
			obs, rew, ter, trunc, info = env.step(action)
			
			proprioception[-1].append(env.current_prop.flatten().tolist())
			actions[-1].append(action.tolist())
			rewards[-1].append(float(rew))
			env.current_render.save(base_path + f'img_{episode}_{step}.png')
			
			tot_rewards[-1] += rew
			tot_success[-1] = (info['success'] == 1) or tot_success[-1]
			
			if ter or trunc:
				# --- POPUP LOGIC ---
				env.reset()
				if auto_accept:
					keep_episode = True
				else:
					print(f"\nEpisode {episode} finished in {step} steps.")
					print(f"Reward: {tot_rewards[-1]:.2f}")
					print(f"Success: {tot_success[-1]}")
					answer = input("Keep this episode data? [y/n]: ").strip().lower()
					keep_episode = answer in ('y', 'yes')
				
				if keep_episode:
					# Keep the data, prep the next episode normally
					obs, info = env.reset()
					if i < n_sample:
						episode += 1
						step = 0
						proprioception.append([env.current_prop.flatten().tolist()])
						actions.append([])
						rewards.append([])
						tot_rewards.append(0)
						tot_success.append(False)
						env.current_render.save(base_path + f'img_{episode}_{step}.png')
				else:
					# Discard the data: Rollback sample counter
					i -= step
					
					# Delete images saved during this bad episode
					for s in range(step + 1):
						img_path = base_path + f'img_{episode}_{s}.png'
						if os.path.exists(img_path):
							os.remove(img_path)
					
					# Reset environment and overwrite current lists 
					obs, info = env.reset()
					step = 0
					actions[-1] = []
					rewards[-1] = []
					proprioception[-1] = [env.current_prop.flatten().tolist()]
					tot_rewards[-1] = 0
					tot_success[-1] = False
					env.current_render.save(base_path + f'img_{episode}_{step}.png')

	env.close()
	# Failsafe: Cleanup if the loop terminated precisely on an empty initialized episode
	if len(actions[-1]) == 0 and len(actions) > 1:
		actions.pop()
		rewards.pop()
		proprioception.pop()
		tot_rewards.pop()
		tot_success.pop()

	with open(action_path, "w") as f:
		json.dump(
			{ "actions": actions, "reward": rewards, "proprioception": proprioception },
			f, indent=4
		)
		
	return tot_rewards, tot_success

if __name__ == "__main__":
	vq   = VQVAE(CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, 0.25, best_device(), True)
	lstm = LSTMQuantized(vq, best_device(), CURRENT_ENV['a_size'], PROP_SIZE, HIDDEN_DIM)
	generate_data_interactive(vq, lstm, 150, True, 3)
	exit()
	from random import randint
	if 'MUJOCO_GL' not in os.environ:
		os.environ['MUJOCO_GL'] = 'egl'
	SMOOTH = True if SMOOTH > 0 else False
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTH, False, False)
	#lstm = load_transformer(CURRENT_ENV, vq, best_device(), EMB_SIZE, MAX_SEQ_LEN, NUM_HEADS, NUM_LAYERS, DROPOUT, False, False)
	env = SoftWrapEnv(vq, lstm)
	observation, _ = env.reset()
	frames = []
	frames.append(env.render().rotate(180))
	done = False
	total_reward = 0
	step_count = 0
	agent = PPO.load(CURRENT_ENV['models'] + 'agent' + f'{EXP_ID}', env)
	while not done:
		if randint(0, 9) < -1:
			action = env.action_space.sample()  # random action
		else:
			action, _states = agent.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"Step {step_count} Reward: {reward} | action: {action} | obs: {observation}")
		if(info['success'] == 1):
			print(f'Win!! Total Reward: {total_reward}')
			#break
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