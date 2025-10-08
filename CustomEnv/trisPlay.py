from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tris import TrisEnv


if __name__ == "__main__":
	env = TrisEnv()
	model = PPO.load("ppo_tris", env=env)
	obs, _ = env.reset()
	
	n_steps = 20
	for step in range(n_steps):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, trunc, info = env.step(action, interactive=True)
		env.render()
		if done:
			# Note that the VecEnv resets automatically
			# when a done signal is encountered
			print("Episode Ended!", "reward=", reward)
			break