from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV, PPO_MODEL
from environments.pseudo_dream import PseudoDreamEnv


if __name__ == "__main__":
	# Load the trained model
	model = PPO.load(CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip")
	
	# Create environment with rendering
	env = Monitor(PseudoDreamEnv(CURRENT_ENV, render_mode="human"))
	
	# Run rollout
	obs, _ = env.reset()
	done = False
	total_reward = 0
	
	time = 0
	while not done:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		print(f"Observation {time}: {obs.sum():.2f}, Reward: {reward:.2f}")
		env.render()
		total_reward += reward
		done = terminated or truncated
		time += 1

	print(f"Episode reward: {total_reward:.2f}")
	env.close()
