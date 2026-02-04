import gymnasium as gym
from PIL import Image
from tqdm import tqdm
import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CAR_RACING

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
	img = renderer.render()
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
	env = gym.make('CarRacing-v3', 
				render_mode='rgb_array',
				)
	renderer = env
	proprioception = [[]]
	actions = [[]]
	rewards = [[]]
	obs, info = env.reset()
	proprioception[-1].append([0])
	episode = 0
	step = 0
	get_img(renderer, size).save(f"data/car_racing/imgs/img_{episode}_{step}.png")
	for i in tqdm(range(n_samples), desc="Generating transitions", unit="transition"):
		step += 1
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		proprioception[-1].append([0])
		get_img(renderer, size).save(f"data/car_racing/imgs/img_{episode}_{step}.png")
		actions[-1].append(action.tolist())
		rewards[-1].append(float(reward))
		if terminated or truncated:
			obs, info = env.reset()
			if i < n_samples - 1:
				episode += 1
				step = 0
				proprioception.append([])
				proprioception[-1].append([0])
				get_img(renderer, size).save(f"data/car_racing/imgs/img_{episode}_{step}.png")
				actions.append([])
				rewards.append([])

	env.close()
	return actions, rewards, proprioception

if __name__ == "__main__":
	# Save the dataset
	if not os.path.exists("data/car_racing/imgs/"):
		os.makedirs("data/car_racing/imgs/")
	
	# Example of creating a dataset of transitions
	print("Creating transition dataset for CarRacing-v3 environment")
	actions, rewards, proprioception = gather_data(n_samples=100000, size=(64, 64))
	print(f"Generated {len(actions)} actions, and {len(rewards)} rewards.")


	if not os.path.exists("data/car_racing/"):
		os.makedirs("data/car_racing/")
	with open("data/car_racing/action_reward_data.json", "w") as f:
		json.dump(
			{
				"actions": actions,
				"reward": rewards,
				"proprioception": proprioception
			},
			f,
			indent=4
		)
