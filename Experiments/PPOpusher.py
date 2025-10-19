import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from evaluator import PeriodicEvalCallback
import sys
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import RescaleAction


if len(sys.argv) != 2 or sys.argv[1] not in ["train", "load"]:
	print("Usage: python PPOpusher.py [train|load]")
	sys.exit(1)

mode = sys.argv[1]

eval_env = Monitor(gym.make('Pusher-v5'))
if mode == "train":
	train_env = Monitor(RescaleAction(gym.make('Pusher-v5', reward_control_weight=0.03, reward_dist_weight=1.5), -1.0, 1.0))
	model = PPO(MlpPolicy, train_env,
			 #learning_rate=4e-4,
			 n_steps=2048,
			 clip_range=0.2,
			 gae_lambda=0.9,
			 batch_size=128,
			 use_sde=False)# other option was terrible

	eval_freq = 20_000
	n_eval_episodes = 5
	callback = PeriodicEvalCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=1)
	total_steps = 1_000_000 # long training does not improve either the variance or the score
	model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)
	model.save("./pusherPPO")

	callback.plot_curve()
else:
	model = PPO.load("./pusherPPO")
	record_env = RecordVideo(gym.make('Pusher-v5', render_mode='rgb_array'),
								video_folder="./videos",
								name_prefix="pusher")
	obs, info = record_env.reset()
	terminated = False
	truncated = False
	while not (terminated or truncated):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = record_env.step(action)
	record_env.close()
final_mean, final_std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final evaluation: mean={final_mean:.2f} std={final_std:.2f}")
	