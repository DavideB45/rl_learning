import os
import sys
import argparse
import platform
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from env_wrapper import MetaWrapEnv
from helpers import best_device
from global_var import *
from impala_cnn import ImpalaCNN


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on visual Meta-World tasks.")
	parser.add_argument(
		"--model-path",
		type=str,
		default="ppo_metaworld_vision.zip",
		help="Path to the trained PPO model (.zip).",
	)
	parser.add_argument(
		"--vec-normalize-path",
		type=str,
		default="vec_normalize.pkl",
		help="Path to saved VecNormalize statistics (optional, loaded if found).",
	)
	parser.add_argument(
		"--n-episodes",
		type=int,
		default=10,
		help="Number of evaluation episodes to run.",
	)
	parser.add_argument(
		"--deterministic",
		action="store_true",
		default=True,
		help="Use deterministic policy actions (greedy mean) during evaluation.",
	)
	parser.add_argument(
		"--stochastic",
		action="store_false",
		dest="deterministic",
		help="Use stochastic actions sampled from the Gaussian policy.",
	)
	parser.add_argument(
		"--record-video",
		action="store_true",
		default=True,
		help="Record and save evaluation episodes as video.",
	)
	parser.add_argument(
		"--no-video",
		action="store_false",
		dest="record_video",
		help="Disable video recording.",
	)
	parser.add_argument(
		"--video-path",
		type=str,
		default="eval_ppo.mp4",
		help="Output path for recorded video (.mp4 or .gif).",
	)
	parser.add_argument(
		"--fps",
		type=int,
		default=25,
		help="Frames per second for saved video.",
	)
	parser.add_argument(
		"--freeze-target",
		action="store_true",
		default=False,
		help="Freeze target button/object in a fixed location across all episodes.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=1000,
		help="Base random seed for evaluation episodes.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# Configure headless rendering backend
	if 'MUJOCO_GL' not in os.environ:
		if platform.system() == 'Darwin':
			os.environ['MUJOCO_GL'] = 'glfw'
		else:
			os.environ['MUJOCO_GL'] = 'egl'

	device = best_device()
	print("=" * 60)
	print(f"PPO Model Evaluation - Meta-World [{CURRENT_ENV['env_name']}]")
	print(f"Device: {device}")
	print(f"Model Path: {args.model_path}")
	print(f"Episodes: {args.n_episodes} | Deterministic: {args.deterministic} | Freeze Target: {args.freeze_target}")
	print("=" * 60)

	# Verify model file exists
	if not os.path.exists(args.model_path):
		# Also check without .zip extension
		if os.path.exists(args.model_path + ".zip"):
			args.model_path = args.model_path + ".zip"
		else:
			raise FileNotFoundError(
				f"Model file '{args.model_path}' not found! Please check the path or train a model first."
			)

	# 1. Build evaluation environment
	def make_eval_env():
		env = MetaWrapEnv(
			env_dict=CURRENT_ENV,
			frame_stack=FRAME_STACK,
			grayscale=GRAYSCALE,
			channels_first=CHANNELS_FIRST,
			action_repeat=(ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
		)
		if args.freeze_target:
			try:
				env.env.unwrapped._freeze_rand_vec = True
			except Exception:
				pass
		return Monitor(env, info_keywords=("is_success",))

	eval_env = DummyVecEnv([make_eval_env])

	# 2. Load running normalization statistics if available
	if os.path.exists(args.vec_normalize_path):
		print(f"Loading VecNormalize statistics from '{args.vec_normalize_path}'...")
		eval_env = VecNormalize.load(args.vec_normalize_path, eval_env)
		# Freeze running statistics during evaluation
		eval_env.training = False
		# Ensure reported rewards are unnormalized environment returns
		eval_env.norm_reward = False
	else:
		print("No VecNormalize statistics file found. Evaluating with raw environment rewards.")

	# 3. Load trained PPO agent
	print(f"Loading PPO policy from '{args.model_path}'...")
	custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0}
	agent = PPO.load(
		args.model_path,
		env=eval_env,
		device=device,
		custom_objects=custom_objects,
	)

	# 4. Evaluation Loop
	episode_rewards = []
	episode_lengths = []
	episode_successes = []
	video_frames = []

	raw_env = eval_env.envs[0]

	for ep in range(args.n_episodes):
		# Seed evaluation episodes for reproducibility
		eval_env.seed(args.seed + ep)
		obs = eval_env.reset()

		ep_reward = 0.0
		ep_length = 0
		ep_success = False
		done = False

		# Capture initial frame if recording
		if args.record_video:
			frame = raw_env.render()
			if frame is not None:
				video_frames.append(frame)

		while not done:
			action, _ = agent.predict(obs, deterministic=args.deterministic)
			obs, reward, dones, infos = eval_env.step(action)

			ep_reward += float(reward[0])
			ep_length += 1
			done = dones[0]

			info = infos[0]
			if info.get("is_success", False) or float(info.get("success", 0.0)) > 0.0:
				ep_success = True

			if args.record_video:
				frame = raw_env.render()
				if frame is not None:
					video_frames.append(frame)

		episode_rewards.append(ep_reward)
		episode_lengths.append(ep_length)
		episode_successes.append(1.0 if ep_success else 0.0)

		status = "SUCCESS" if ep_success else "FAILED "
		print(
			f"Episode {ep + 1:2d}/{args.n_episodes} | "
			f"Result: [{status}] | "
			f"Reward: {ep_reward:7.2f} | "
			f"Length: {ep_length:3d}"
		)

	# 5. Aggregate Results
	success_rate = (sum(episode_successes) / len(episode_successes)) * 100.0
	mean_reward = float(np.mean(episode_rewards))
	std_reward = float(np.std(episode_rewards))
	mean_length = float(np.mean(episode_lengths))

	print("=" * 60)
	print("EVALUATION SUMMARY")
	print(f"Total Episodes : {args.n_episodes}")
	print(f"Success Rate   : {success_rate:.1f}% ({int(sum(episode_successes))}/{args.n_episodes} successful)")
	print(f"Mean Reward    : {mean_reward:.2f} +/- {std_reward:.2f}")
	print(f"Min / Max Rew  : {min(episode_rewards):.2f} / {max(episode_rewards):.2f}")
	print(f"Mean Length    : {mean_length:.1f} steps")
	print("=" * 60)

	# 6. Save Video Recording
	if args.record_video and len(video_frames) > 0:
		os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
		print(f"Saving evaluation recording to '{args.video_path}' ({len(video_frames)} frames @ {args.fps} FPS)...")
		imageio.mimsave(args.video_path, video_frames, fps=args.fps)
		print(f"Video saved successfully to '{args.video_path}'")

	eval_env.close()


if __name__ == "__main__":
	main()
