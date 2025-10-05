import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

t = 0
done = False
while not done:
	env.render()
	action = env.action_space.sample()
	obs,reward,done,truncated,info = env.step(action)
	t += 1
	if done:
		print("Done after {} steps".format(t))
		break
env.close()