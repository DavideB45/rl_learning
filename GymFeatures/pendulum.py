import gymnasium as gym
from EnvWrppers import NormalizeActionWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

pendulum = Monitor(gym.make('Pendulum-v1'))
pendulum = NormalizeActionWrapper(pendulum)
pendulum = DummyVecEnv([lambda: pendulum])
# For now it is just a training algorithm as any other
# I'm just learning the sintax for gymnasium
model = A2C(MlpPolicy, pendulum, verbose=0,
            #learning_rate=7e-4,
            n_steps=8,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            gae_lambda=0.9,
            use_sde=True       
)

# evaluate non trained model
trials = 10
avg, std = evaluate_policy(model, pendulum, n_eval_episodes=trials)
print(f"Evaluation over {trials} trials after 0 steps:\nAvg ret:\t{avg}\nstd:  \t{std}")

steps = 400000
model.learn(steps, progress_bar=True)
avg, std = evaluate_policy(model, pendulum, n_eval_episodes=trials)
print(f"Evaluation over {trials} trials after {steps} steps:\nAvg ret:\t{avg}\nstd:  \t{std}")


model.save("./pendulumA2C")

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