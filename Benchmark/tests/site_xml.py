import gymnasium as gym
import metaworld
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV

if __name__ == "__main__":
	env = gym.make('Meta-World/MT1', env_name=CURRENT_ENV['env_name'],
				render_mode='human', camera_id="5",
		)
	env.reset()
	env.render()
	terminated = False
	truncated = False
	while not (terminated or truncated):
		action = env.action_space.sample()
		action[3] = 1
		action[2] = -1
		action[1] = -1
		obs, reward, terminated, truncated, info = env.step(action)
		print(obs)
		env.render()
		time.sleep(0.1)
		
	env.close()