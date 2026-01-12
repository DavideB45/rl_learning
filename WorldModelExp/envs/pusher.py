import gymnasium as gym
from PIL import Image
from tqdm import tqdm
import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import PUSHER

# This file contains functions to create a dataset of transitions
# (state, action, next_state) for training world models.


def get_img(renderer, size=(64, 64)) -> Image.Image:
	'''
	Renders the current frame of the environment and resizes it.
	Args:
		env: gym environment
		size: desired size of the image
	Returns:
		Image.Image: resized image
	'''
	renderer.camera_id = 2
	img = renderer.render(render_mode='rgb_array')
	img = Image.fromarray(img)
	img = img.resize(size)
	return img

def gather_data(n_samples=1000, size=(64, 64)):
	'''
	Create a dataset of transitions from the pusher environment
	n_samples: number of samples to generate
	size: size of the images to generate
	returns: list of images
	'''
	env = gym.make('Pusher-v5', 
				render_mode='rgb_array',
				default_camera_config=PUSHER['default_camera_config'],
				)
	renderer = env.env.env.env.mujoco_renderer
	images = [[]]
	actions = [[]]
	rewards = [[]]
	obs, info = env.reset()
	images[-1].append(get_img(renderer, size))
	for i in tqdm(range(n_samples), desc="Generating transitions", unit="transition"):
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		images[-1].append(get_img(renderer, size))
		actions[-1].append(action.tolist())
		rewards[-1].append(float(reward))
		if terminated or truncated:
			obs, info = env.reset()
			if i < n_samples - 1:
				images.append([])
				images[-1].append(get_img(renderer, size))
				actions.append([])
				rewards.append([])

	# Close renderer safely
	if hasattr(env, "mujoco_renderer") and env.mujoco_renderer is not None:
		env.mujoco_renderer.close()
		env.mujoco_renderer = None

	#env.close()
	return images, actions, rewards

if __name__ == "__main__":
	
	# Example of creating a dataset of transitions
	print("Creating transition dataset for Pusher-v5 environment")
	images, actions, rewards = gather_data(n_samples=1000, size=(64, 64))
	print(f"Generated {len(images)} images, {len(actions)} actions, and {len(rewards)} rewards.")
	
	# Save the dataset
	if not os.path.exists("data/pusher/imgs/"):
		os.makedirs("data/pusher/imgs/")
	for episode in range(len(images)):
		for i, img in enumerate(images[episode]):
			img.save(f"data/pusher/imgs/img_{episode}_{i}.png")

	if not os.path.exists("data/pusher/"):
		os.makedirs("data/pusher/")
	with open("data/pusher/action_reward_data.json", "w") as f:
		json.dump(
			{
				"actions": actions,
				"reward": rewards
			},
			f,
			indent=4
		)
