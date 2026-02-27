import gymnasium as gym
import time

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

	times = {
	"collecting_time": 11479.811615228653,
	"vq_training_time": 1414.4060266017914,
	"lstm_training_time": 2443.2115354537964,
	"dataset_generation_time": 1635.7053062915802,
	"agent_training_time": 3350.7150461673737,
	"evaluation_time": 776.5424153804779
	} # 20 iterazioni
	times = {
	"collecting_time": 15034.367637634277,
	"vq_training_time": 2924.7899508476257,
	"lstm_training_time": 4407.352098226547,
	"dataset_generation_time": 3200.922375202179,
	"agent_training_time": 4832.109834432602,
	"evaluation_time": 999.9526515007019
	} # 30 iterazioni
	tot = 0
	for key in times:
		tot += times[key]
		print(f"{key} : {time.strftime('%H:%M:%S', time.gmtime(times[key]))}")
	print(f"tot : {time.strftime('%H:%M:%S', time.gmtime(tot))}")