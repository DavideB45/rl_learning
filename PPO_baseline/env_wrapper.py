import numpy as np
import gymnasium as gym
from gymnasium import spaces
import metaworld
from stable_baselines3 import PPO
import platform

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers import best_device
from global_var import *

class MetaWrapEnv(gym.Env):
	"""
	This environment is a wrapper of the real environment used to change
	the observation space from the position of robots and objects to
	the image space (RGB pixels).
	"""

	def __init__(self):
		'''
		initialization of the wrapper
		'''
		super(MetaWrapEnv, self).__init__()

		self.env = gym.make('Meta-World/MT1', env_name=CURRENT_ENV['env_name'],
				render_mode='rgb_array', camera_id=CURRENT_ENV['camera_id'],
				width = CURRENT_ENV['render_size'], height = CURRENT_ENV['render_size'])
		
		self.env.env.env.env.env.env.env.env.model.cam_pos[2][:]=[0.75, 0.075, 0.7]
		
		self.action_space = spaces.Box(
			low=-1, high=1, 
			shape=(4,), 
			dtype=np.float32
		)
		
		# Use the rendered RGB image as the observation.
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(CURRENT_ENV['render_size'], CURRENT_ENV['render_size'], 3),
			dtype=np.uint8,
		)
		
		self.current_render = None

	def get_img(self) -> np.ndarray:
		'''
		Renders the current frame of the environment as a NumPy array.
		'''
		img = self.env.render()
		# Ensure it is a numpy array of uint8 to match the observation space
		return np.array(img, dtype=np.uint8)
	
	def reset(self, seed=None, options=None):
		'''
		Reset the environment
		'''
		super().reset(seed=seed, options=options)
		_, _ = self.env.reset(seed=seed)
		img = self.get_img()
		self.current_render = img
		return img, {"success": 0, "info": "reset"}

	def step(self, action) -> tuple:
		'''
		Step in the environment
		'''
		_, reward, terminated, truncated, info = self.env.step(action)
		
		if not (terminated or truncated) and ACTION_REPEAT:
			_, reward_, terminated, truncated, info = self.env.step(action)
			reward += reward_
			
		img = self.get_img()
		self.current_render = img
		
		return (
			img, 
			float(reward), 
			bool(terminated), 
			bool(truncated), 
			info 
		)
	
	def render(self):
		return self.current_render
		
	def close(self):
		self.env.close()

if __name__ == "__main__":
	if 'MUJOCO_GL' not in os.environ:
		if platform.system() == 'Darwin':  # macOS
			os.environ['MUJOCO_GL'] = 'glfw'
		else:  # Linux / servers
			os.environ['MUJOCO_GL'] = 'egl'
		
	# 1. Initialize the environment
	env = MetaWrapEnv()
	
	# 2. Instantiate PPO with "CnnPolicy" so it builds the NatureCNN feature extractor
	# You can pass the best_device() if it returns a standard torch device string (e.g., 'cuda:0')
	agent = PPO(
		policy="CnnPolicy", 
		env=env, 
		verbose=1,
		device=best_device(), 
	)
	
	# 3. Train the agent using learn()
	print("Starting PPO training on images...")
	agent.learn(total_timesteps=N_ROUNDS*(500 if not ACTION_REPEAT else 250), progress_bar=True)
	
	# Optional: Save your model after training
	agent.save("ppo_metaworld_vision")