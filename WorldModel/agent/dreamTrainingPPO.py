from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import torch

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

if __name__ == "__main__":
	# We suppose we altrady have:
	# - a trained PPO model saved in CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip"
	# - a MDRNN and VAE model trained and saved in the data_dir
	# Those can be obtained running makePPO.py after training the world models

	if not os.path.exists(CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip"):
		raise FileNotFoundError(f"PPO model not found at {CURRENT_ENV['data_dir'] + PPO_MODEL + '.zip'}. Please train the PPO model first.")

	# Make some new experience in the pseudo-dream environment
	experience_env = PseudoDreamEnv(CURRENT_ENV, render_mode="rgb_array")
	policy = PPO.load(CURRENT_ENV['data_dir'] + PPO_MODEL + ".zip", env=experience_env)
	images, history = make_experience(experience_env, policy, n_steps=1000)

	# Now use the experience to finetune the VAE and MDNRNN models
	vae = VAE()
	vae.load_state_dict(torch.load(CURRENT_ENV['data_dir'] + VAE_MODEL, map_location=torch.device('cpu')))
	train_vae(vae_=vae, images=images, epochs=5)

	mdrnn = MDNRNN()
	mdrnn.load_state_dict(torch.load(CURRENT_ENV['data_dir'] + MDRNN_MODEL, map_location=torch.device('cpu')))
	train_mdrnn(mdrnn_=mdrnn, data_=history, epochs=10)

	# Save the finetuned models
	torch.save(vae.state_dict(), VAE_MODEL)
	torch.save(mdrnn.state_dict(), MDRNN_MODEL)
