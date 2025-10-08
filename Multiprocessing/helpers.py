import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed


# The multiprocessing implementation requires a function that
# can be called inside the process to instantiate a gym env
# this is a simpler version of stable_baselines3.common.env_util.make_vec_env
def make_env(env_id, rank, seed=0):
	"""
	Utility function for multiprocessed env.

	:param env_id: (str) the environment ID
	:param seed: (int) the inital seed for RNG
	:param rank: (int) index of the subprocess
	"""

	def _init():
		env = gym.make(env_id)
		# use a seed for reproducibility
		# Important: use a different seed for each environment
		# otherwise they would generate the same experiences
		env.reset(seed=seed + rank)
		return env

	set_random_seed(seed)
	return _init