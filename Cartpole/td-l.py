import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

for i_episode in range(3):
	obs = env.reset()
	t = 0
	done = False
	while not done:
		env.render()
		action = None
		while action not in [0, 1]:
			key = input("Press 0 (left) or 1 (right) for action: ")
			try:
				action = int(key)
			except ValueError:
				action = None
		obs,reward,done,truncated,info = env.step(action)
		t += 1
		if done:
			print("Done after {} steps".format(t))
			break
env.close()