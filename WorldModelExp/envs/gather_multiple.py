import argparse
import gymnasium as gym
from PIL import Image
from typing import Tuple

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import PUSHER
from tqdm import tqdm


def get_img(renderer, size=(64, 64)) -> Tuple[Image.Image, Image.Image, Image.Image]:
	renderer.camera_id = 0  # Switch to 'cam_above'
	img1 = renderer.render(render_mode='rgb_array')
	img1 = Image.fromarray(img1).resize(size)
	renderer.camera_id = 1  # Switch to 'cam_side'
	img2 = renderer.render(render_mode='rgb_array')
	img2 = Image.fromarray(img2).resize(size)
	renderer.camera_id = 2  # Switch to 'cam_front'
	img3 = renderer.render(render_mode='rgb_array')
	img3 = Image.fromarray(img3).resize(size)
	return img1, img2, img3

def run(env_id="Pusher-v5", episodes=5, max_steps=100, seed=0):
	render_mode = "rgb_array"
	env = gym.make(env_id, render_mode=render_mode)
	renderer = env.env.env.env.mujoco_renderer
	cameras = ["cam_above 0", "cam_side 1", "cam_front 2"]
	above_img = []
	side_img = []
	front_img = []
	try:
		for ep in tqdm(range(episodes), desc="Episodes"):
			obs, info = env.reset(seed=seed + ep)
			for t in range(max_steps):
				action = env.action_space.sample()
				obs, reward, terminated, truncated, info = env.step(action)
				img1, img2, img3 = get_img(renderer, size=(64, 64))
				above_img.append(img1)
				side_img.append(img2)
				front_img.append(img3)
				if terminated or truncated:
					break
		print(f"Collected {len(above_img)} images from 'cam_above' and {len(side_img)} images from 'cam_side'.")
		
		# for i in range(len(above_img)):
		# 	above_img[i] = cv2.cvtColor(above_img[i], cv2.COLOR_RGB2BGR)
		# 	side_img[i] = cv2.cvtColor(side_img[i], cv2.COLOR_RGB2BGR)
			
		print(f'shape of above_img: {above_img[0].size}, shape of side_img: {side_img[0].size}')
		print(f"Color range check - above_img[0]: min {min(above_img[0].getdata())}, max {max(above_img[0].getdata())}")
		# save images in CURRENT_ENV data_dir
		if not os.path.exists("data/pusher/multi_img/"):
			os.makedirs("data/pusher/multi_img/")
		for i in range(len(above_img)):
			above_img[i].save(f"data/pusher/multi_img/above_img_{i:06d}.png")
			side_img[i].save(f"data/pusher/multi_img/side_img_{i:06d}.png")
			front_img[i].save(f"data/pusher/multi_img/front_img_{i:06d}.png")
		print("Images saved in data/pusher/multi_img/")
	finally:
		env.close()


if __name__ == "__main__":
	p = argparse.ArgumentParser(description="Run Pusher-v5 with gymnasium")
	p.add_argument("--env", default="Pusher-v5", help="Environment id")
	p.add_argument("--episodes", type=int, default=1, help="Number of episodes")
	p.add_argument("--steps", type=int, default=100, help="Max steps per episode")
	args = p.parse_args()

	run(env_id=args.env, episodes=args.episodes, max_steps=args.steps)