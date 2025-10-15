import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from evaluator import PeriodicEvalCallback
import sys


if len(sys.argv) != 2 or sys.argv[1] not in ["train", "load"]:
	print("Usage: python SACwalker2D.py [train|load]")
	sys.exit(1)

mode = sys.argv[1]

eval_env = Monitor(gym.make('Walker2d-v5'))
if mode == "train":
	train_env = Monitor(gym.make('Walker2d-v5'))
	model = SAC(MlpPolicy, train_env)

	eval_freq = 20_000
	n_eval_episodes = 5
	callback = PeriodicEvalCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=1)
	total_steps = 1_000_000
	model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)
	model.save("./walkerSAC")

	callback.plot_curve()
else:
	model = SAC.load("./walkerSAC")
	env = gym.make('Walker2d-v5', render_mode="human")
	obs, info = env.reset()
	done = False
	while not done:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, trunc, info = env.step(action)
	env.close()
final_mean, final_std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final evaluation: mean={final_mean:.2f} std={final_std:.2f}")
