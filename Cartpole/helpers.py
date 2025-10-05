from stable_baselines3.common.base_class import BaseAlgorithm
import tqdm

# Actually this already exists as a built-in function in stable-baselines3
# but it can be useful to see how it works
def evaluate_policy(model: BaseAlgorithm, env, n_eval_episodes: int = 10):
    """
    Evaluate a RL agent

    :param model: (BaseAlgorithm) The RL Agent
    :param env: (Gym environment) The environment to evaluate on
    :param n_eval_episodes: (int) Number of episodes to evaluate the agent
    :return: (float, float) Mean reward for the last n_eval_episodes and standard deviation
    """
    all_episode_rewards = []
    for _ in tqdm.tqdm(range(n_eval_episodes), desc="Evaluating"):
        episode_rewards = 0.0
        done = False
        obs, info = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
        all_episode_rewards.append(episode_rewards)
    mean_reward = sum(all_episode_rewards) / n_eval_episodes
    std_reward = (sum((x - mean_reward) ** 2 for x in all_episode_rewards) / n_eval_episodes) ** 0.5
    return mean_reward, std_reward