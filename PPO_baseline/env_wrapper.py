import os
import sys
import platform
from collections import deque
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import metaworld
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers import best_device
from global_var import *
from impala_cnn import ImpalaCNN

class MetaWrapEnv(gym.Env):
	"""
	Environment wrapper for Meta-World tasks that converts the observation
	space to visual representations with frame stacking (e.g. 3 consecutive
	black-and-white frames) and action repeat.
	"""
	metadata = {"render_modes": ["rgb_array"]}

	def __init__(
		self,
		env_dict: dict = CURRENT_ENV,
		frame_stack: int = FRAME_STACK,
		grayscale: bool = GRAYSCALE,
		channels_first: bool = CHANNELS_FIRST,
		action_repeat: int = (ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
	):
		"""
		Initialization of the visual wrapper.

		:param env_dict: Configuration dictionary for the Meta-World environment.
		:param frame_stack: Number of consecutive frames to stack (default: 3).
		:param grayscale: Whether to convert RGB frames to single-channel grayscale (default: True).
		:param channels_first: Whether output shape is (C, H, W) for PyTorch (default: True).
		:param action_repeat: Number of physics simulation steps to repeat each action (default: 2).
		"""
		super(MetaWrapEnv, self).__init__()

		self.env_dict = env_dict
		self.frame_stack = max(1, frame_stack)
		self.grayscale = grayscale
		self.channels_first = channels_first
		self.action_repeat = max(1, action_repeat)
		self.render_size = env_dict['render_size']

		self.env = gym.make(
			'Meta-World/MT1',
			env_name=env_dict['env_name'],
			render_mode='rgb_array',
			camera_id=env_dict['camera_id'],
			width=self.render_size,
			height=self.render_size,
		)

		try:
			self.env.env.env.env.env.env.env.env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
		except Exception:
			pass

		self.action_space = spaces.Box(
			low=-1, high=1,
			shape=(env_dict.get('a_size', 4),),
			dtype=np.float32,
		)

		# Determine observation shape:
		# For grayscale + 3 frames stacked: shape is (3, H, W) if channels_first else (H, W, 3)
		# For RGB + 3 frames stacked: shape is (9, H, W) if channels_first else (H, W, 9)
		num_channels = self.frame_stack if self.grayscale else self.frame_stack * 3
		if self.channels_first:
			obs_shape = (num_channels, self.render_size, self.render_size)
		else:
			obs_shape = (self.render_size, self.render_size, num_channels)

		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=obs_shape,
			dtype=np.uint8,
		)

		self.frames = deque(maxlen=self.frame_stack)
		self.current_render = None
		self.episode_success = 0.0

	def get_img(self) -> np.ndarray:
		"""
		Renders the current RGB frame of the environment as a uint8 NumPy array.
		"""
		img = self.env.render()
		return np.array(img, dtype=np.uint8)

	def _process_frame(self, rgb_img: np.ndarray) -> np.ndarray:
		"""
		Converts raw RGB image into grayscale if enabled.
		"""
		if self.grayscale:
			return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
		return rgb_img

	def _get_obs(self) -> np.ndarray:
		"""
		Stacks buffered frames along the appropriate channel dimension.
		"""
		if self.grayscale:
			# Each frame in self.frames is shape (H, W)
			if self.channels_first:
				return np.stack(self.frames, axis=0)
			return np.stack(self.frames, axis=-1)
		else:
			# Each frame in self.frames is shape (H, W, 3)
			if self.channels_first:
				frames_chw = [np.transpose(f, (2, 0, 1)) for f in self.frames]
				return np.concatenate(frames_chw, axis=0)
			return np.concatenate(self.frames, axis=-1)

	def reset(self, seed=None, options=None):
		"""
		Reset the environment and initialize the frame stack buffer with initial observations.
		"""
		super().reset(seed=seed, options=options)
		_, _ = self.env.reset(seed=seed)

		self.episode_success = 0.0
		raw_img = self.get_img()
		self.current_render = raw_img
		processed = self._process_frame(raw_img)

		self.frames.clear()
		for _ in range(self.frame_stack):
			self.frames.append(processed)

		obs = self._get_obs()
		return obs, {"success": 0.0, "is_success": False, "info": "reset"}

	def step(self, action) -> tuple:
		"""
		Step in the environment with action repeat and frame stacking.
		Tracks task success across the entire episode.
		"""
		total_reward = 0.0
		terminated = False
		truncated = False
		info = {}

		for _ in range(self.action_repeat):
			_, reward, term, trunc, step_info = self.env.step(action)
			total_reward += float(reward)
			terminated = bool(term)
			truncated = bool(trunc)
			info = step_info

			# Track if task was successfully completed at any point during the episode
			if float(step_info.get("success", 0.0)) > 0.0:
				self.episode_success = 1.0

			if terminated or truncated:
				break

		raw_img = self.get_img()
		self.current_render = raw_img
		processed = self._process_frame(raw_img)
		self.frames.append(processed)

		# Propagate is_success (required by SB3 to record rollout/success_rate)
		info["is_success"] = bool(self.episode_success > 0.0)
		info["success"] = float(self.episode_success)

		obs = self._get_obs()
		return (
			obs,
			total_reward,
			terminated,
			truncated,
			info,
		)

	def render(self):
		return self.current_render

	def close(self):
		try:
			self.env.close()
		except Exception:
			pass


def make_env(
	env_dict: dict = CURRENT_ENV,
	frame_stack: int = FRAME_STACK,
	grayscale: bool = GRAYSCALE,
	channels_first: bool = CHANNELS_FIRST,
	action_repeat: int = (ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
) -> gym.Env:
	"""
	Factory helper that creates a single MetaWrapEnv wrapped with SB3's Monitor.
	"""
	env = MetaWrapEnv(
		env_dict=env_dict,
		frame_stack=frame_stack,
		grayscale=grayscale,
		channels_first=channels_first,
		action_repeat=action_repeat,
	)
	return Monitor(env, info_keywords=("is_success",))


def linear_schedule(initial_value: float, final_value: float = 1e-5):
	"""
	Linear learning rate schedule that decays from initial_value to final_value.
	"""
	def func(progress_remaining: float) -> float:
		return final_value + progress_remaining * (initial_value - final_value)
	return func


def make_vec_envs(
	env_dict: dict = CURRENT_ENV,
	n_envs: int = N_ENVS,
	seed: int = 42,
	frame_stack: int = FRAME_STACK,
	grayscale: bool = GRAYSCALE,
	channels_first: bool = CHANNELS_FIRST,
	action_repeat: int = (ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
	normalize_reward: bool = NORMALIZE_REWARD,
	use_subproc: bool = False,
):
	"""
	Creates vectorized environments with seeding, Monitor wrappers,
	and optional running reward normalization (VecNormalize).
	"""
	def _init(rank: int):
		def _thunk():
			env = MetaWrapEnv(
				env_dict=env_dict,
				frame_stack=frame_stack,
				grayscale=grayscale,
				channels_first=channels_first,
				action_repeat=action_repeat,
			)
			env.action_space.seed(seed + rank)
			return Monitor(env, info_keywords=("is_success",))
		return _thunk

	env_fns = [_init(i) for i in range(n_envs)]

	# On macOS Darwin, DummyVecEnv avoids GLFW OpenGL context crashes across subprocesses.
	# On headless Linux with EGL, SubprocVecEnv can be used.
	if n_envs > 1 and use_subproc:
		venv = SubprocVecEnv(env_fns)
	else:
		venv = DummyVecEnv(env_fns)

	if normalize_reward:
		venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

	return venv


if __name__ == "__main__":
	if 'MUJOCO_GL' not in os.environ:
		if platform.system() == 'Darwin':  # macOS
			os.environ['MUJOCO_GL'] = 'glfw'
		else:  # Linux / servers
			os.environ['MUJOCO_GL'] = 'egl'

	device = best_device()
	print(f"Device: {device}")
	print(f"Observation setup: Frame stack={FRAME_STACK}, Grayscale={GRAYSCALE}, Channels first={CHANNELS_FIRST}")
	print(f"Parallel Envs: {N_ENVS}, Reward Normalization: {NORMALIZE_REWARD}")

	# 1. Initialize vectorized environments (DummyVecEnv on macOS to avoid GLFW multithreading issues)
	use_subproc = (platform.system() != 'Darwin')
	env = make_vec_envs(
		env_dict=CURRENT_ENV,
		n_envs=N_ENVS,
		frame_stack=FRAME_STACK,
		grayscale=GRAYSCALE,
		channels_first=CHANNELS_FIRST,
		action_repeat=(ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
		normalize_reward=NORMALIZE_REWARD,
		use_subproc=use_subproc,
	)
	print(f"Observation space: {env.observation_space}")
	print(f"Action space: {env.action_space}")

	# 2. Configure feature extractor (Impala CNN vs Nature CNN)
	policy_kwargs = {}
	if USE_IMPALA:
		print(f"Backbone: ImpalaCNN (residual blocks, depths={IMPALA_DEPTHS}, features_dim={PPO_FEATURES_DIM})")
		policy_kwargs = dict(
			features_extractor_class=ImpalaCNN,
			features_extractor_kwargs=dict(
				features_dim=PPO_FEATURES_DIM,
				depths=IMPALA_DEPTHS,
			),
		)
	else:
		print("Backbone: NatureCNN (default SB3)")

	# 3. Instantiate PPO with linear LR schedule and tuned hyperparameters for image RL (Idea 3 & 6)
	lr_schedule = linear_schedule(PPO_LR, final_value=PPO_MIN_LR)

	agent = PPO(
		policy="CnnPolicy",
		env=env,
		policy_kwargs=policy_kwargs,
		learning_rate=lr_schedule,
		n_steps=PPO_N_STEPS,
		batch_size=PPO_BATCH_SIZE,
		n_epochs=PPO_N_EPOCHS,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		ent_coef=PPO_ENT_COEF,
		verbose=1,
		device=device,
		tensorboard_log="./tensorboard_logs/",
	)

	# 4. Train the agent using learn()
	total_steps = N_ROUNDS * (500 if not ACTION_REPEAT else 250)
	print(f"Starting PPO training on images for {total_steps} timesteps...")
	agent.learn(total_timesteps=total_steps, progress_bar=True)

	# 4. Save model and normalization statistics
	agent.save("ppo_metaworld_vision")
	print("Model saved to ppo_metaworld_vision.zip")
	if NORMALIZE_REWARD and isinstance(env, VecNormalize):
		env.save("vec_normalize.pkl")
		print("VecNormalize statistics saved to vec_normalize.pkl")

	env.close()