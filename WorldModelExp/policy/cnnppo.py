import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np

def make_env():
	env = gym.make("Pusher-v5", render_mode="rgb_array")
	env = ResizeObservation(env, shape=(64, 64))   # Resize frames to 64x64
	env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale, keep channel dim
	env = FrameStack(env, num_stack=4)             # Stack 4 frames for temporal context
	return env

# Vectorize the environment (required by SB3)
vec_env = DummyVecEnv([make_env])
vec_env = VecTransposeImage(vec_env)  # Transpose to (C, H, W) format expected by SB3

# Create the PPO agent with CnnPolicy
model = PPO(
	policy="CnnPolicy",
	env=vec_env,
	learning_rate=2.5e-4,
	n_steps=512,
	batch_size=64,
	n_epochs=4,
	gamma=0.99,
	gae_lambda=0.95,
	clip_range=0.1,
	verbose=1,
	tensorboard_log="./ppo_pusher_tensorboard/"
)

print("Starting training...")
model.learn(total_timesteps=500_000)

# Save the model
model.save("ppo_pusher_cnn")
print("Model saved!")

# --- Evaluation ---
eval_env = DummyVecEnv([make_env])
eval_env = VecTransposeImage(eval_env)

obs = eval_env.reset()
total_reward = 0
done = False

for _ in range(1000):
	action, _ = model.predict(obs, deterministic=True)
	obs, reward, done, info = eval_env.step(action)
	total_reward += reward[0]
	if done[0]:
		break

print(f"Total reward during evaluation: {total_reward:.2f}")
eval_env.close()