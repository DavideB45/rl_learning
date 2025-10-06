import gymnasium as gym
from EnvWrppers import NormalizeActionWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.a2c import A2C


model = A2C.load("./pendulumA2C")

pendulum = env = Monitor(gym.make('Pendulum-v1', render_mode="human"))
pendulum = NormalizeActionWrapper(pendulum)
obs, info = pendulum.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = pendulum.step(action)
    pendulum.render()
    if terminated or truncated:
        obs, info = pendulum.reset()
pendulum.close()