import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER

from envs.simulator import PusherDreamEnv
from envs.wrapper import PusherWrapEnv, generate_data

if __name__ == '__main__':
	SMOOTH = False
	KL = False
	vq = load_vq_vae(PUSHER, 64, 16, 4, True, SMOOTH, best_device())
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), 1024, SMOOTH, True, KL)
	dream_env = PusherDreamEnv(vq, lstm, 10, 200000)
	model = PPO(MlpPolicy, dream_env, verbose=0)

# STEPS
# 1 - gather some amount of data
# 2 - train a vector quantizer variational autoencoder
# 3 - train an lstmc
# 4 - a loop of some length begins
# 4.1 - train a PPO in the dream
# 4.2 - use the PPO to obtain data from a wrapped env
# 4.3 - tune the vq-vae
# 4.4 - tune the lstmc (more than the vq-vae)
# 5 fine

