import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from PIL import Image
import cv2
import numpy as np
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 3.0,
    "azimuth": -90.0,
    "elevation": -20.0,
}

class PixelOnlyWrapper(gym.ObservationWrapper):
    """Replaces the state observation with the rendered pixel frame."""
    def __init__(self, env):
        super().__init__(env)
        env.reset()
        sample = env.render()
        h, w, c = sample.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=sample.dtype
        )

    def observation(self, obs):
        return self.env.render()


def make_env():
    env = gym.make(
        "Pusher-v5",
        render_mode="rgb_array",
        default_camera_config=DEFAULT_CAMERA_CONFIG,
    )
    env = PixelOnlyWrapper(env)                    # state → (H, W, 3) pixels
    env = ResizeObservation(env, shape=(64, 64))   # → (64, 64, 3)
    env = GrayscaleObservation(env, keep_dim=True) # → (64, 64, 1)
    return env

# Build the vec env pipeline using SB3 wrappers only
vec_env = DummyVecEnv([make_env])
vec_env = VecTransposeImage(vec_env)   # (1, 64, 64) — works now, no FrameStack yet
vec_env = VecFrameStack(vec_env, n_stack=4)  # → (4, 64, 64)

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
    tensorboard_log="./ppo_pusher_tensorboard/",
)

print("Starting training...")
#model.learn(total_timesteps=500_000, progress_bar=True)
#model.save("ppo_pusher_cnn")
model.load("ppo_pusher_cnn")
print("Model saved!")

# --- Evaluation ---
eval_env = DummyVecEnv([make_env])
eval_env = VecTransposeImage(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)

obs = eval_env.reset()
total_reward = 0

for _ in range(1000):
    action = eval_env.action_space.sample()
    #action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward[0]
    img = eval_env.render()
    img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)
    image_resized = image.resize((512, 512), Image.NEAREST)
    cv2.imshow('DreamEnv', np.array(image_resized))
    cv2.waitKey(100)
    if done[0]:
        break

print(f"Total reward during evaluation: {total_reward:.2f}")
eval_env.close()