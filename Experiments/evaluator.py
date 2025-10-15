import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class PeriodicEvalCallback(BaseCallback):
    """
    Evaluate the agent every eval_freq timesteps on eval_env and store results.
    """
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps = []
        self.means = []
        self.stds = []

    def _on_step(self) -> bool:
        # self.num_timesteps is the number of timesteps so far
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
                                                      n_eval_episodes=self.n_eval_episodes,
                                                      return_episode_rewards=False,
                                                      warn=False)
            self.timesteps.append(self.num_timesteps)
            self.means.append(mean_reward)
            self.stds.append(std_reward)
            if self.verbose:
                print(f"[Eval @ {self.num_timesteps}] mean: {mean_reward:.2f} std: {std_reward:.2f}")
        return True

    def plot_curve(self):
        timesteps = np.array(self.timesteps)
        means = np.array(self.means)
        stds = np.array(self.stds)

        plt.figure(figsize=(8,5))
        plt.plot(timesteps, means, label='Mean reward (eval)')
        plt.fill_between(timesteps, means - stds, means + stds, alpha=0.2, label='Std')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean reward')
        plt.title('Periodic evaluation during training')
        plt.grid(True)
        plt.legend()
        plt.show()
