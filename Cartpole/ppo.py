import gymnasium as gym

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO

env = gym.make('CartPole-v1')

model = PPO(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=30000)

# Enjoy trained agent
env = gym.make('CartPole-v1', render_mode="human")
obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()