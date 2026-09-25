"""
tests/test_real_policy.py

Evaluate a trained policy (VQ-VAE + LSTM world-model encoder + PPO agent) on the
REAL robot. This does NOT gather training data, and does not push/pull anything
to the server — it only loads whatever checkpoints are currently sitting in
CURRENT_ENV['models'] and runs a handful of episodes so you can see how the
trained policy actually behaves, with a reward/success summary at the end.

Usage:
    python tests/test_real_policy.py --episodes 5
    python tests/test_real_policy.py --episodes 3 --stochastic --video
"""

import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import numpy as np
from stable_baselines3.ppo import PPO

from envs.wrapper import SoftWrapEnv
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from global_var import *


def parse_args():
	p = argparse.ArgumentParser(description="Evaluate the trained policy on the real robot.")
	p.add_argument("--episodes", type=int, default=5, help="number of test episodes to run")
	p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the deterministic policy mean")
	p.add_argument("--video", action="store_true", help="save a .mp4 of every test episode")
	return p.parse_args()


def main():
	args = parse_args()
	deterministic = not args.stochastic

	SMOOTHING = True if SMOOTH > 0 else False
	device = best_device()

	print("[test_real_policy] Loading VQ-VAE + LSTM world model...")
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, device)
	lstm = load_lstm_quantized(CURRENT_ENV, vq, device, HIDDEN_DIM, SMOOTHING, cl=False, kl=False)

	agent_path = CURRENT_ENV['models'] + 'agent' + f'{EXP_ID}'
	print(f"[test_real_policy] Loading PPO agent from {agent_path}")
	agent = PPO.load(agent_path)

	env = SoftWrapEnv(vq, lstm)
	rewards = []
	successes = []
	try:
		for ep in range(args.episodes):
			obs, _ = env.reset()
			if args.video:
				# armed after reset(), not before: reset() itself calls the
				# underlying save_now(), which would otherwise immediately clear
				# this flag before any frame of the episode gets captured
				env.save_next_episode(f'test_{ep}')

			total_reward = 0.0
			success = False
			done = False
			step = 0
			while not done:
				action, _ = agent.predict(obs, deterministic=deterministic)
				# render()/frame capture already happens inside step() via get_img(),
				# no need to call env.render() again here
				obs, reward, terminated, truncated, info = env.step(action)
				total_reward += reward
				success = success or (info.get('success', 0) == 1)
				done = terminated or truncated
				step += 1

			rewards.append(total_reward)
			successes.append(success)
			print(f"[test_real_policy] Episode {ep + 1}/{args.episodes}: "
				  f"steps={step} reward={total_reward:.3f} success={success}")
	finally:
		env.close()

	rewards = np.array(rewards)
	print("\n[test_real_policy] === Summary ===")
	print(f"  episodes      : {len(rewards)}")
	print(f"  mean reward   : {rewards.mean():.3f}")
	print(f"  std reward    : {rewards.std():.3f}")
	print(f"  success rate  : {100.0 * np.mean(successes):.1f}%")


if __name__ == "__main__":
	main()
