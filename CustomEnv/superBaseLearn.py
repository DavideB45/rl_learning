import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from superBase import GoLeftEnv


if __name__ == "__main__":
	vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(grid_size=10))
	model = PPO("MlpPolicy", vec_env, verbose=0)
	model.learn(total_timesteps=100, progress_bar=True)

	# Evaluate the trained agent
	env = Monitor(GoLeftEnv(grid_size=10))
	# NOTE: on my machine this line sometimes is super slow
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
	print(f"Mean reward: {mean_reward} +/- {std_reward}")

	obs, _ = env.reset()
	
	n_steps = 20
	for step in range(n_steps):
		action, _ = model.predict(obs, deterministic=True)
		print(f"Step {step + 1}")
		print("Action: ", action)
		obs, reward, done, trunc, info = env.step(action)
		print("obs=", obs, "reward=", reward, "done=", done)
		env.render()
		if done:
			# Note that the VecEnv resets automatically
			# when a done signal is encountered
			print("Goal reached!", "reward=", reward)
			break