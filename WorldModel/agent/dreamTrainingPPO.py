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
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.train_vae import train_vae
from models.train_mdrnn import train_mdrnn

EXPERIENCE_STEPS = 40000
TUNING_EPOCHS_VAE = 5
TUNING_EPOCHS_MDRNN = 10
TUNING_PPO_TIMESTEPS = 250000
ITERATIONS = 3

if __name__ == "__main__":
	# We suppose we altrady have:
	# - a trained PPO model saved in CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip"
	# - a MDRNN and VAE model trained and saved in the data_dir
	# Those can be obtained running makePPO.py after training the world models

	if not os.path.exists(CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip"):
		raise FileNotFoundError(f"PPO model not found at {CURRENT_ENV['data_dir'] + PPO_MODEL + '.zip'}. Please train the PPO model first.")

	for iteration in range(ITERATIONS):
		print(f"\n=== Iteration {iteration + 1}/{ITERATIONS} ===")
		iteration_start = time.time()
		
		# Make some new experience in the pseudo-dream environment
		start_time = time.time()
		experience_env = PseudoDreamEnv(CURRENT_ENV, render_mode="rgb_array")
		policy = PPO.load(CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip", env=experience_env)
		images, history = make_experience(experience_env, policy, n_steps=EXPERIENCE_STEPS)
		experience_time = time.time() - start_time
		print(f"Experience generation time: {experience_time:.2f} seconds ({experience_time/60:.2f} minutes)")

		# Now use the experience to finetune the VAE and MDNRNN models
		start_time = time.time()
		vae = VAE()
		vae.load_state_dict(torch.load(CURRENT_ENV['data_dir'] + VAE_MODEL, map_location=torch.device('cpu')))
		train_vae(vae_=vae, images=images, epochs=TUNING_EPOCHS_VAE)
		vae_time = time.time() - start_time
		print(f"VAE training time: {vae_time:.2f} seconds ({vae_time/60:.2f} minutes)")
		
		start_time = time.time()
		mdrnn = MDNRNN()
		mdrnn.load_state_dict(torch.load(CURRENT_ENV['data_dir'] + MDRNN_MODEL, map_location=torch.device('cpu')))
		train_mdrnn(mdrnn_=mdrnn, data_=history, epochs=TUNING_EPOCHS_MDRNN, seq_len=15)
		mdrnn_time = time.time() - start_time
		print(f"MDRNN training time: {mdrnn_time:.2f} seconds ({mdrnn_time/60:.2f} minutes)")
		
		torch.save(vae.state_dict(), CURRENT_ENV['data_dir'] + VAE_MODEL)
		torch.save(mdrnn.state_dict(), CURRENT_ENV['data_dir'] + MDRNN_MODEL)

		# Finally, we can retrain the PPO model in the dream environment with the finetuned world models
		start_time = time.time()
		dream_env = DreamEnv(CURRENT_ENV, render_mode="rgb_array")
		eval_callback = EvalCallback(experience_env, 
							   eval_freq=15000,
							   best_model_save_path=CURRENT_ENV['data_dir'],
							   n_eval_episodes=5
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
		del policy, vae, mdrnn
		del images, history