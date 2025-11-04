import json
import gymnasium as gym
from PIL import Image	
import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from tqdm import tqdm

from models.vae import VAE
from torchvision.transforms import ToTensor

# This file contains functions to create transition datasets
# using the vae model and a random policy

def append_information(image:Image, action, reward, last_state, vae:VAE, history:dict):
	'''
	Append the encoded image, action, reward, last_state to the history
	image: input image (of the environment step)
	action: action taken
	reward: reward received
	last_state: boolean indicating if it's the last state of the episode
	vae: VAE model to encode the image
	history: dictionary containing the history of transitions
	returns: updated history dictionary
	'''
	if len(history) == 0:
		history.append({'mu':[], 'log_var':[], 'action':[], 'reward':[], 'last_state':[]})
	with torch.no_grad():
		mu, log_var = vae.encode(ToTensor()(image).unsqueeze(0))
	history[-1]['mu'].append(mu.squeeze(0).detach().cpu().tolist())
	history[-1]['log_var'].append(log_var.squeeze(0).detach().cpu().tolist())
	history[-1]['action'].append(action.tolist())
	history[-1]['reward'].append(reward)
	history[-1]['last_state'].append(last_state)
	if last_state:
		history.append({'mu':[], 'log_var':[], 'action':[], 'reward':[], 'last_state':[]})
	return history

def make_transition_data(env_name='Pendulum-v1', n_samples=1000, size=(64, 64), vae_model_path=None, default_camera_config=None):
	if CURRENT_ENV['special_call'] is not None:
		CURRENT_ENV['special_call']()
	if default_camera_config is not None:
		env = gym.make(env_name, render_mode='rgb_array', default_camera_config=default_camera_config)
	else:
		env = gym.make(env_name, render_mode='rgb_array')
	_, _ = env.reset()
	history = []
	vae = VAE(
		latent_dim=CURRENT_ENV['z_size'],
	)
	vae.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
	vae.eval()
	for _ in tqdm(range(n_samples), desc="Generating experience", unit="moments"):
		action = env.action_space.sample()
		if env_name == 'CarRacing-v3':
			action[1] = max(action[1], 0.5)  # accelerate
			action[2] = min(action[2], 0.3)  # low brake
		img = env.render()
		img = Image.fromarray(img)
		img = img.resize(size)
		_, reward, terminated, truncated, _ = env.step(action)
		if terminated or truncated:
			_, _ = env.reset()
		history = append_information(img, action, reward, terminated or truncated, vae, history)
	env.close()
	return history

if __name__ == "__main__":
	# Example of creating a dataset of transitions
	vae_model_path = CURRENT_ENV['vae_model']
	history = make_transition_data(env_name=CURRENT_ENV['env_name'], 
						n_samples=200000, 
						size=(64, 64), 
						vae_model_path=vae_model_path,
						default_camera_config=CURRENT_ENV['default_camera_config']
						)
	print(f"Generated transition data from {CURRENT_ENV['env_name']} environment")
	# Save history to disk
	json.dump(history, open(CURRENT_ENV['transitions'], 'w'), indent=4)