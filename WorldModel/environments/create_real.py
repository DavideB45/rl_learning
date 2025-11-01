import gymnasium as gym
from PIL import Image
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from tqdm import tqdm


# This file contains the functions to use the real environment
# and the functions used to create the datasets for training world models
# Can create a dataset of images

def make_img_data(env_name='Pendulum-v1',n_samples=1000, size=(64, 64), default_camera_config=None):
	'''
	Create a dataset of images from the env_name environment
	n_samples: number of samples to generate
	size: size of the images to generate
	policy: function that takes an observation and returns an action
	returns: list of images
	'''
	if CURRENT_ENV['special_call'] is not None:
		CURRENT_ENV['special_call']()
	if default_camera_config is not None:
		env = gym.make(env_name, render_mode='rgb_array', default_camera_config=default_camera_config)
	else:
		env = gym.make(env_name, render_mode='rgb_array')
	images = []
	_, _ = env.reset()
	for _ in tqdm(range(n_samples), desc="Generating images", unit="img"):
		action = env.action_space.sample()
		# if in car racing press the gas
		if env_name == 'CarRacing-v3':
			action[1] = max(action[1], 0.5)  # accelerate
			action[2] = min(action[2], 0.3)  # low brake
		_, _, terminated, truncated, _ = env.step(action)
		img = env.render()
		img = Image.fromarray(img)
		img = img.resize(size)
		images.append(img)
		if terminated or truncated:
			_, _ = env.reset()
	env.close()
	return images

def make_first_frame(env_dict=CURRENT_ENV, size=(64, 64)) -> tuple[Image.Image, gym.spaces.Space]:
	if env_dict['special_call'] is not None:
		env_dict['special_call']()
	env = gym.make(env_dict['env_name'], render_mode='rgb_array')
	action_space = env.action_space
	_, _ = env.reset()
	img = env.render()
	img = Image.fromarray(img)
	img = img.resize(size)
	env.close()
	return img, action_space

if __name__ == "__main__":
	# Example of creating a dataset of images
	images = make_img_data(env_name=CURRENT_ENV['env_name'], 
						n_samples=100000, 
						size=(64, 64), 
						default_camera_config=CURRENT_ENV['default_camera_config']
						)
	print(f"Generated {len(images)} images from {CURRENT_ENV['env_name']} environment")
	# Save images to disk
	for i, img in enumerate(images):
		img.save(f"{CURRENT_ENV['img_dir']}{i:05d}.png")

