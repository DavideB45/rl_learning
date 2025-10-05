import gymnasium as gym

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.ppo import PPO

env = gym.make('CartPole-v1')

model = PPO(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=10000)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()