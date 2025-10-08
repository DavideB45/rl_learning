from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tris import TrisEnv


if __name__ == "__main__":
	env = Monitor(TrisEnv())
	model = PPO.load("ppo_tris", env=env)
	obs, _ = env.reset()
	
	n_steps = 20
	for step in range(n_steps):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, trunc, info = env.step(action, interactive=True)
		print("reward=", reward, "done=", done)
		env.render()
		if done:
			# Note that the VecEnv resets automatically
			# when a done signal is encountered
			print("Goal reached!", "reward=", reward)
			break