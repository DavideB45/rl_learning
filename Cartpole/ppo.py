import gymnasium as gym

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO

from helpers import evaluate_policy

env = gym.make('CartPole-v1')

model = PPO(MlpPolicy, env, verbose=0)

# Evaluate untrained agent
mean_reward, std_reward = evaluate_policy(model, env)
print(f"Mean reward (untrained): {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent
model.learn(total_timesteps=20000, progress_bar=True)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

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