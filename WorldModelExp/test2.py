import gymnasium as gym

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO


if __name__ == '__main__':
	env = gym.make_vec('CartPole-v1', num_envs=3)
	print('---- Action Space ----')
	print(env.action_space.shape)
	print(env.single_action_space.shape)
	print('---- Observation ----')
	print(env.observation_space.shape)
	print(env.single_observation_space.shape)