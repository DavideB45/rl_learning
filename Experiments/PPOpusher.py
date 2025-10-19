import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import sys
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import RescaleAction
from itertools import product
import json


if len(sys.argv) != 2 or sys.argv[1] not in ["train", "load"]:
	print("Usage: python PPOpusher.py [train|load]")
	sys.exit(1)

LEARNING_RATE = [3e-4, 4e-4]
CLIP_RANGE = [0.1, 0.2, 0.3]
GAE_LAMBDA = [0.9, 0.95, 0.98]
N_EPOCS = [4, 10]
N_STEPS = [2048]

results = []

mode = sys.argv[1]
eval_env = Monitor(RescaleAction(gym.make('Pusher-v5', max_episode_steps=180), -1.0, 1.0))
if mode == "train":
	train_env = RescaleAction(gym.make('Pusher-v5'), -1.0, 1.0)
	total_iterations = len(LEARNING_RATE) * len(CLIP_RANGE) * len(GAE_LAMBDA) * len(N_STEPS)
	current_iteration = 0
	best_mean = -float('inf')
	# Hyperparameter tuning
	for lr, cr, gl, ns in product(LEARNING_RATE, CLIP_RANGE, GAE_LAMBDA, N_STEPS):
		current_iteration += 1
		print(f"Hyperparameter tuning iteration {current_iteration}/{total_iterations}")
		print(f"Training with lr={lr}, clip_range={cr}, gae_lambda={gl}, n_steps={ns} ()")
		model = PPO(MlpPolicy, train_env,
					learning_rate=lr,
					n_steps=ns,
					clip_range=cr,
					gae_lambda=gl,
					batch_size=128,
					use_sde=False)# other option was terrible
		eval_freq = 20_000
		n_eval_episodes = 5
		total_steps = 1_000_000 # long training does not improve either the variance or the score
		model.learn(total_timesteps=total_steps, progress_bar=True)
		final_mean, final_std = evaluate_policy(model, eval_env, n_eval_episodes=10)
		results.append((lr, cr, gl, ns, final_mean, final_std))
		print(f"Result: lr={lr}, clip_range={cr}, gae_lambda={gl}, n_steps={ns} -> mean={final_mean:.2f} std={final_std:.2f}")
		if final_mean > best_mean:
			best_mean = final_mean
			model.save("./pusherPPO")

		with open("results.json", "w") as f:
			json.dump(results, f, indent=4)
else:
	model = PPO.load("./pusherPPO")
	DEFAULT_CAMERA_CONFIG = {
		"trackbodyid": -1,   # no specific body tracking
		"distance": 3.0,     # distance from the agent
		"azimuth": -90.0,    # rotate camera to front of the agent
		"elevation": -20.0,  # lower angle for a front view
	}
	record_env = RecordVideo(
		RescaleAction(
			gym.make('Pusher-v5', render_mode='rgb_array', max_episode_steps=180,
				default_camera_config=DEFAULT_CAMERA_CONFIG,
			), -1.0, 1.0
		),
		video_folder="./videos",
		name_prefix="pusher"
	)
	obs, info = record_env.reset()
	terminated = False
	truncated = False
	while not (terminated or truncated):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = record_env.step(action)
	record_env.close()
final_mean, final_std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final evaluation: mean={final_mean:.2f} std={final_std:.2f}")
	