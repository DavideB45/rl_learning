from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from tris import TrisEnv


if __name__ == "__main__":
	env = Monitor(TrisEnv())
	model = PPO("MlpPolicy", env, verbose=0)
	model.learn(total_timesteps=10000, progress_bar=True)

	# Evaluate the trained agent
	model.save("ppo_tris")
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
	print(f"Mean reward: {mean_reward} +/- {std_reward}")

	obs, _ = env.reset()
	
	n_steps = 20
	for step in range(n_steps):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, trunc, info = env.step(action)
		print("reward=", reward, "done=", done)
		env.render()
		if done:
			# Note that the VecEnv resets automatically
			# when a done signal is encountered
			print("Goal reached!", "reward=", reward)
			break