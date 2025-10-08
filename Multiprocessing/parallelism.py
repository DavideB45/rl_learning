# Basically vectorized environments make use of multiprocessing to run
# multiple environments in parallel. This is useful for RL algorithms that
# require a lot of data, as it can significantly speed up the data collection.
# Also, it can help to stabilize training by providing a more diverse set of experiences.

# SubprocVecEnv is a class that creates multiple environments in separate processes.
# DummyVecEnv is a class that creates multiple environments in the same process.

import numpy as np
import gymnasium as gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


# on my computer results are like 1/3 better, it is not much but can be useful
if __name__ == "__main__":
	# We do some benchmarking 
	env_id = "Pendulum-v1"
	PROCESSES_TO_TEST = [1, 4]
	NUM_EXPERIMENTS = 3  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
	TRAIN_STEPS = 100000
	EVAL_EPS = 20
	eval_env = Monitor(gym.make(env_id))
	
	reward_averages = []
	reward_std = []
	training_times = []
	total_procs = 0
	for n_procs in PROCESSES_TO_TEST:
		total_procs += n_procs
		print(f'Running for n_procs = {n_procs}')
		# Here we are using only one process even for n_env > 1
		# this is equivalent to DummyVecEnv([make_env(env_id, i + total_procs) for i in range(n_procs)])
		train_env = make_vec_env(env_id, n_envs=n_procs)

		rewards = []
		times = []

		for experiment in range(NUM_EXPERIMENTS):
			# it is recommended to run several experiments due to variability in results
			train_env.reset()
			model = PPO("MlpPolicy", train_env, verbose=0)
			start = time.time()
			model.learn(total_timesteps=TRAIN_STEPS)
			times.append(time.time() - start)
			mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
			rewards.append(mean_reward)

		train_env.close()
		reward_averages.append(np.mean(rewards))
		reward_std.append(np.std(rewards))
		training_times.append(np.mean(times))

		print(f"Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
		print(f"Average training time: {np.mean(times):.2f} seconds")