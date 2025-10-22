import gymnasium as gym
from PIL import Image
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import PENDULUM


# This file contains the functions to use the real environment
# and the functions used to create the datasets for training world models
# Can create a dataset of images
# TODO: add functions to create datasets of states and actions
# It uses the gymnasium Pendulum-v1 environment

def make_img_data(env_name='Pendulum-v1',n_samples=1000, size=(64, 64)):
	'''
	Create a dataset of images from the Pendulum-v1 environment
	n_samples: number of samples to generate
	size: size of the images to generate
	policy: function that takes an observation and returns an action
	returns: list of images
	'''
	env = gym.make('Pendulum-v1', render_mode='rgb_array')
	images = []
	_, _ = env.reset()
	for _ in range(n_samples):
		action = env.action_space.sample()
		_, _, terminated, truncated, _ = env.step(action)
		img = env.render()
		img = Image.fromarray(img)
		img = img.resize(size)
		images.append(img)
		if terminated or truncated:
			_, _ = env.reset()
	env.close()
	return images

if __name__ == "__main__":
	# Example of creating a dataset of images
	images = make_img_data(env_name=PENDULUM['env_name'], n_samples=5000, size=(64, 64))
	print(f"Generated {len(images)} images from {PENDULUM['env_name']} environment")
	# Save images to disk
	for i, img in enumerate(images):
		img.save(f"{PENDULUM['img_dir']}{i:05d}.png")

