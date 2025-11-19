import time
import argparse
import gymnasium as gym
import inspect

#!/usr/bin/env python3
"""
Simple runner for the "Pusher-v5" environment using gymnasium.

Save as: /Users/davideborghini/Documents/GitHub/rl_learning/WorldModelExp/envs/mutliview_test.py
Run: python mutliview_test.py
"""


def run(env_id="Pusher-v5", episodes=5, max_steps=500, seed=0, render=True):
	render_mode = "rgb_array" if render else None
	env = gym.make(env_id, render_mode=render_mode)
	print(dir(env.env.env.env))
	print()
	renderer = env.env.env.env.mujoco_renderer
	print(dir(renderer))
	print()
	print(inspect.getfullargspec(renderer.render))
	print(inspect.getfullargspec(renderer._set_cam_config))
	#print
	#exit()
	cameras = ["cam_above 0", "cam_side 1"]
	above_img = []
	side_img = []
	try:
		for ep in range(episodes):
			obs, info = env.reset(seed=seed + ep)
			print(f"Episode {ep+1}/{episodes} started. Observation type: {type(obs)}")
			for t in range(max_steps):
				action = env.action_space.sample()
				obs, reward, terminated, truncated, info = env.step(action)
				renderer.camera_id = 0  # Switch to 'cam_above'
				above_img.append(renderer.render(render_mode=render_mode))
				renderer.camera_id = 1  # Switch to 'cam_side'
				side_img.append(renderer.render(render_mode=render_mode))
				#env.render()
				if terminated or truncated:
					print(f"Episode {ep+1} finished after {t+1} steps. reward={reward:.3f}")
					break
		print(f"Collected {len(above_img)} images from 'cam_above' and {len(side_img)} images from 'cam_side'.")
		#Save videos
		import cv2
		import matplotlib.pyplot as plt
		fig, axes = plt.subplots(1, 2, figsize=(12, 5))
		axes[0].imshow(above_img[30])
		axes[0].set_title('Camera Above')
		axes[0].axis('off')
		axes[1].imshow(side_img[30])
		axes[1].set_title('Camera Side')
		axes[1].axis('off')
		plt.show()
		for i in range(len(above_img)):
			above_img[i] = cv2.resize(above_img[i], (64, 64))
			side_img[i] = cv2.resize(side_img[i], (64, 64))
			above_img[i] = cv2.cvtColor(above_img[i], cv2.COLOR_RGB2BGR)
			side_img[i] = cv2.cvtColor(side_img[i], cv2.COLOR_RGB2BGR)
			
		print(f'shape of above_img: {above_img[0].shape}, shape of side_img: {side_img[0].shape}')
		print(f"Color range check - above_img[0]: min {above_img[0].min()}, max {above_img[0].max()}")
		height, width, _ = above_img[0].shape
		above_video = cv2.VideoWriter('pusher_above.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
		side_video = cv2.VideoWriter('pusher_side.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
		for img in above_img:
			above_video.write(img)
		for img in side_img:
			side_video.write(img)
		above_video.release()
		side_video.release()
		print("Videos saved: 'pusher_above.mp4' and 'pusher_side.mp4'")
	finally:
		env.close()


if __name__ == "__main__":
	p = argparse.ArgumentParser(description="Run Pusher-v5 with gymnasium")
	p.add_argument("--env", default="Pusher-v5", help="Environment id")
	p.add_argument("--episodes", type=int, default=1, help="Number of episodes")
	p.add_argument("--steps", type=int, default=100, help="Max steps per episode")
	p.add_argument("--no-render", action="store_true", help="Disable rendering")
	args = p.parse_args()

	run(env_id=args.env, episodes=args.episodes, max_steps=args.steps, render=not args.no_render)