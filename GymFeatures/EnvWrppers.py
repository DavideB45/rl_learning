import gymnasium as gym
import numpy as np

# This also actually already exists as a function in gym 
# gymnasium.wrappers.TimeLimit
class TimeLimit(gym.Wrapper):
	"""
	:param env: (gym.Env) Gym environment that will be wrapped
	:param max_steps: (int) Max number of steps per episode
	"""

	def __init__(self, env, max_step=100):
		super(TimeLimit, self).__init__(env) # this way we can call self.env later
		self.max_step = max_step
		self.current_steps = 0

	def reset(self, **kwargs): #function to reimplement
		"""
		Reset the envirionment
		"""
		self.current_steps = 0
		return self.env.reset( **kwargs)
	
	def step(self, action): # function to reimplement
		"""
		Do an action in the environment
		"""
		self.current_steps += 1
		obs, reward, terminated, truncated, info = self.env.step(action)
		if self.current_steps > self.max_step:
			truncated = True
		return obs, reward, terminated, truncated, info

class NormalizeActionWrapper(gym.Wrapper):
	
	def __init__(self, env:gym.Env):
		action_space = env.action_space
		# spaces.Box is the kind of continuos action
		assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"

		self.low, self.high = action_space.low, action_space.high
		# We define an action space in the range we want with the correct shape
		# This will not modify what the real environment should get, just what the action space is 
		env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
		super(NormalizeActionWrapper, self).__init__(env)

	def rescale_action(self, scaled_action):
		"""
		Rescale the action from [-1, 1] to [low, high]
		(no need for symmetric action space)
		:param scaled_action: (np.ndarray)
		:return: (np.ndarray)
		"""
		return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))
	
	def step(self, action):
		"""
		Do an action in the environment
		:param action: ([float] or int) Action taken by the agent
		:retun: (np.ndarray, float, bool, bool, dict) observation, reward, final state?, truncated?, additional informations
		"""
		# The action receied is in normalized space
		action = self.rescale_action(action)
		obs, reward, terminated, truncated, info = self.env.step(action)
		return obs, reward, terminated, truncated, info

if __name__ == '__main__':
	from gymnasium.envs.classic_control.pendulum import PendulumEnv

	#we do like this because otherwise the environment is already wrapped
	penzolo = PendulumEnv()
	penzolo = TimeLimit(penzolo)

	obs, _ = penzolo.reset()
	done = False
	steps = 0
	while not done:
		action = penzolo.action_space.sample()
		_, _, end, trunc, info = penzolo.step(action)
		done = end or trunc
		steps += 1
	
	print(f"Info = {info}\nCounted Steps = {steps}")