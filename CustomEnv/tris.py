import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrisEnv(gym.Env):
	"""
	In this env the agent must learn to play tris (tic-tac-toe)
	the agent plays always 'X' and the opponent plays always 'O'
	the opponent plays randomly
	"""

	# This was a tutorial in colab so there was no gui
	metadata = {"render_modes": ["console"]}

	# Define constants for clearer code
	AGENT = 0
	OPPONENT = 1

	def __init__(self, render_mode="console"):
		super(TrisEnv, self).__init__()
		self.render_mode = render_mode
		self.board = np.full((3, 3), -1)  # -1 means empty cell
		self.current_player = self.AGENT  # agent starts first
		self.action_space = spaces.Discrete(9)
		self.observation_space = spaces.Box(
			low=-1, high=1, shape=(3, 3), dtype=np.int8
		)  # -1 empty, 0 agent, 1 opponent


	def reset(self, seed=None, options=None):
		"""
		Important: the observation must be a numpy array
		:return: (np.array)
		"""
		super().reset(seed=seed, options=options)
		self.current_player = self.AGENT
		self.board[:] = -1
		obs = self.board.copy().astype(np.int8)

		return obs, {}  # empty info dict

	def _playMove(self, player, place) -> bool:
		'''
		Try to play a move in the desired place
		A move can be illegal if the cell is already in use
		or if it is a number greater than 8
		'''
		if place is not None:
			x:int =  place//3
			y = place % 3
			if self.board[x,y] == -1:
				self.board[x,y] = player
				return True
			else:
				return False
		else:  # opponent plays randomly
			empty_cells = np.argwhere(self.board == -1)
			if len(empty_cells) == 0:
				return False
			choice = self.np_random.choice(len(empty_cells))
			x, y = empty_cells[choice]
			self.board[x, y] = 1
			return True
			
	# Check if the game is over (win or draw)
	def _checkWinner(self) -> int:
		'''
		Check if there is a winner in the ugliest way possible
		:return: AGENT if agent wins, OPPONENT if opponent wins, -1 if draw, None if game not over
		'''
		if np.all(self.board[0, :] == self.AGENT) or \
		   np.all(self.board[1, :] == self.AGENT) or \
		   np.all(self.board[2, :] == self.AGENT) or \
		   np.all(self.board[:, 0] == self.AGENT) or \
		   np.all(self.board[:, 1] == self.AGENT) or \
		   np.all(self.board[:, 2] == self.AGENT) or \
		   np.all(np.diag(self.board) == self.AGENT) or \
		   np.all(np.diag(np.fliplr(self.board)) == self.AGENT):
			return self.AGENT
		if np.all(self.board[0, :] == self.OPPONENT) or \
		   np.all(self.board[1, :] == self.OPPONENT) or \
		   np.all(self.board[2, :] == self.OPPONENT) or \
		   np.all(self.board[:, 0] == self.OPPONENT) or \
		   np.all(self.board[:, 1] == self.OPPONENT) or \
		   np.all(self.board[:, 2] == self.OPPONENT) or \
		   np.all(np.diag(self.board) == self.OPPONENT) or \
		   np.all(np.diag(np.fliplr(self.board)) == self.OPPONENT):
			return self.OPPONENT
		if np.all(self.board != -1):
			return -1  # draw
		return None  # game not over

	def step(self, action, interactive=False):
		"""
		here we take a step from the agent and then we do a step from the opponent
		If the action is illegal we give a negative reward and end the game
		:param action: the action of the agent
		:return: observation, reward, terminated, truncated, info
		"""

		if not self._playMove(self.AGENT, action):
			# Illegal move
			return (
				self.board.copy().astype(np.float32),
				-2.0,
				True,
				False, # maybe this can be True
				{"legal_action": False},
			)

		# Check if the game is over
		winner = self._checkWinner()
		if winner is not None:
			return (
				self.board.copy().astype(np.float32),
				1 if winner == self.AGENT else 0, # here either 1 (win) or 0 (draw)
				True,
				False,
				{"legal_action": True},
			)
		
		# Opponent plays
		if interactive:
			self.render()
			while True:
				input_str = input("Enter your move (0-8): ")
				try:
					place = int(input_str)
					if place < 0 or place > 8:
						print("Invalid move. Try again.")
						continue
					if not self._playMove(self.OPPONENT, place):
						print("Cell already occupied. Try again.")
						continue
					break
				except ValueError:
					print("Invalid input. Try again.")
		else:
			self._playMove(self.OPPONENT, None)
		winner = self._checkWinner()
		if winner is not None:
			return (
				self.board.copy().astype(np.float32),
				-1, # here only OPPONENT win is possible
				True,
				False,
				{"legal_action": True},
			)
		
		return (
			self.board.copy().astype(np.float32),
			0.0,
			False,
			False,
			{"legal_action": True},
		)

	def render(self):
		if self.render_mode == "console":
			for row in self.board:
				print("|".join([' ' if x == -1 else ('X' if x == 0 else 'O') for x in row]))
				print("-" * 5)
			print("\n")

	def close(self):
		pass

if __name__ == "__main__":
	env = TrisEnv()
	observation, info = env.reset()
	env.render()
	done = False
	while not done:
		action = env.action_space.sample()  # random action
		observation, reward, terminated, truncated, info = env.step(action)
		env.render()
		done = terminated or truncated
		if done:
			print(f"Game over! Reward: {reward}")
	env.close()