import gymnasium as gym
import metaworld

from PIL import Image
from tqdm import tqdm
import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV, TRANSITIONS
from helpers.data import get_data_path

def get_img(renderer) -> Image.Image:
	'''
	Renders the current frame of the environment and resizes it.
	Args:
		env: gym environment
	Returns:
		Image.Image: resized image
	'''
	img = renderer.render()
	img = Image.fromarray(img)
	return img

def gather_data(n_samples=1000, path=0):
	'''
	Create a dataset of transitions from the pusher environment
	n_samples: number of samples to generate
	size: size of the images to generate
	returns: list of images
	'''
	env = gym.make('Meta-World/MT1', env_name=CURRENT_ENV['env_name'],
				render_mode='rgb_array', camera_id=CURRENT_ENV['camera_id'], width = 128, height = 128,
		)
	
	proprioception = [[]]
	actions = [[]]
	rewards = [[]]
	obs, info = env.reset()
	proprioception[-1].append(obs.tolist()[0:4])
	episode = 0
	step = 0
	get_img(env).save(f"{path}/img_{episode}_{step}.png")
	for i in tqdm(range(n_samples), desc="Generating transitions", unit="transition"):
		step += 1
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		proprioception[-1].append(obs.tolist()[0:4])
		get_img(env).save(f"{path}/img_{episode}_{step}.png")
		actions[-1].append(action.tolist())
		rewards[-1].append(float(reward))
		if terminated or truncated:
			obs, info = env.reset()
			if i < n_samples - 1:
				episode += 1
				step = 0
				proprioception.append([])
				proprioception[-1].append(obs.tolist()[0:4])
				get_img(env).save(f"{path}/img_{episode}_{step}.png")
				actions.append([])
				rewards.append([])

	env.close()
	return actions, rewards, proprioception

if __name__ == "__main__":

	path = get_data_path(CURRENT_ENV['img_dir'], True, 0)
	if not os.path.exists(path):
		os.makedirs(path)
	if not os.path.exists(CURRENT_ENV['models']):
		os.makedirs(CURRENT_ENV['models'])
	
	# Example of creating a dataset of transitions
	print(f"Creating transition dataset for {CURRENT_ENV['env_name']} environment")
	actions, rewards, proprioception = gather_data(n_samples=3000, path=path)
	print(f"Generated {len(actions)} actions, and {len(rewards)} rewards.")


	with open(path + TRANSITIONS, "w") as f:
		json.dump(
			{
				"actions": actions,
				"reward": rewards,
				"proprioception": proprioception
			},
			f,
			indent=4
		)
