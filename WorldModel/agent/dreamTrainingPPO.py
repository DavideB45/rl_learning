import json
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import torch
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from environments.pseudo_dream import PseudoDreamEnv, make_experience
from environments.american_dream import DreamEnv
from global_var import CURRENT_ENV, PPO_MODEL, VAE_MODEL, MDRNN_MODEL
from models.mdnrnn import MDNRNN
from models.train_mdrnn import train_mdrnn

EXPERIENCE_STEPS = 320_000 # gather data requires ~2h
TUNING_EPOCHS_MDRNN = 20 # ~?h
TUNING_PPO_TIMESTEPS = 2_000_000 # ~30 minutes
ITERATIONS = 1

CREATE_EXPERIENCE = False
TRAIN_MDRNN = False
TRAIN_PPO = True

if __name__ == "__main__":
	# We suppose we altrady have:
	# - a trained PPO model saved in CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip"
	# - a MDRNN and VAE model trained and saved in the data_dir
	# Those can be obtained running makePPO.py after training the world models

	if not os.path.exists(CURRENT_ENV['ppo_model'] + ".zip"):
		raise FileNotFoundError(f"PPO model not found at {CURRENT_ENV['ppo_model'] + '.zip'}. Please train the PPO model first.")

	for iteration in range(ITERATIONS):
		print(f"\n=== Iteration {iteration + 1}/{ITERATIONS} ===")
		iteration_start = time.time()
		
		# Make some new experience in the pseudo-dream environment
		start_time = time.time()
		experience_env = PseudoDreamEnv(CURRENT_ENV, render_mode="rgb_array")
		if CREATE_EXPERIENCE:
			policy = PPO.load(CURRENT_ENV['ppo_model'], env=experience_env)
			_, history = make_experience(experience_env, policy, n_steps=EXPERIENCE_STEPS)
			with open(CURRENT_ENV['data_dir'] + "experience.json", "w") as f:
				json.dump(history, f, indent=4)
		elif TRAIN_MDRNN:
			history = None
			with open(CURRENT_ENV['data_dir'] + "experience.json", "r") as f:
				history = json.load(f)
		experience_time = time.time() - start_time
		print(f"Experience generation time: {experience_time:.2f} seconds ({experience_time/60:.2f} minutes)")
		
		if TRAIN_MDRNN:
			start_time = time.time()
			mdrnn = MDNRNN(
				z_size=CURRENT_ENV['z_size'],
				a_size=CURRENT_ENV['a_size'],
				rnn_size=CURRENT_ENV['rnn_size'],
				n_gaussians=CURRENT_ENV['num_gaussians'],
				reward_weight=5.0
			)
			train_mdrnn(mdrnn_=mdrnn, data_=history, epochs=TUNING_EPOCHS_MDRNN, seq_len=100, noise_scale=0.5)
			mdrnn_time = time.time() - start_time
			print(f"MDRNN training time: {mdrnn_time:.2f} seconds ({mdrnn_time/60:.2f} minutes)")
			torch.save(mdrnn.state_dict(), CURRENT_ENV['data_dir'] + MDRNN_MODEL)
			mdrnn.eval()
		else:
			mdrnn = MDNRNN(
				z_size=CURRENT_ENV['z_size'],
				a_size=CURRENT_ENV['a_size'],
				rnn_size=CURRENT_ENV['rnn_size'],
				n_gaussians=CURRENT_ENV['num_gaussians']
			)
			mdrnn.load_state_dict(torch.load(CURRENT_ENV['data_dir'] + MDRNN_MODEL, map_location=torch.device('cpu')))
			mdrnn.eval()

		# Finally, we can retrain the PPO model in the dream environment with the finetuned world models
		start_time = time.time()
		if TRAIN_PPO:
			dream_env = Monitor(DreamEnv(CURRENT_ENV, render_mode="rgb_array"))
			policy = PPO(MlpPolicy, dream_env, learning_rate=3e-4, n_steps=2048, clip_range=0.2,
							gae_lambda=0.95, batch_size=128)
			eval_callback = EvalCallback(experience_env,
						eval_freq=100000, # High number because it is 25x slower than the dream env
						best_model_save_path=CURRENT_ENV['data_dir'],
						n_eval_episodes=3
						)
			policy.learn(total_timesteps=TUNING_PPO_TIMESTEPS, callback=eval_callback, progress_bar=True)
			policy.save(CURRENT_ENV['data_dir'] + PPO_MODEL)
			ppo_time = time.time() - start_time
			print(f"PPO training time: {ppo_time:.2f} seconds ({ppo_time/60:.2f} minutes)")
			print("Finished dream tuning PPO training.")

		iteration_time = time.time() - iteration_start
		print(f"\n--- Iteration {iteration + 1} Total Time: {iteration_time:.2f} seconds ({iteration_time/60:.2f} minutes) ---")

		#clean up
		experience_env.close()
		dream_env.close()
		del experience_env, dream_env
		del policy, mdrnn
		del history