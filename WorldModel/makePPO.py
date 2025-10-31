from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from global_var import CURRENT_ENV, PPO_MODEL
from environments.pseudo_dream import PseudoDreamEnv
from stable_baselines3.common.monitor import Monitor


if __name__ == "__main__":
	env = Monitor(PseudoDreamEnv(CURRENT_ENV, render_mode="none"))
	model = PPO(MlpPolicy, env, learning_rate=3e-4, n_steps=2048, clip_range=0.2,
				gae_lambda=0.95, batch_size=128, use_sde=False)
	eval_env = Monitor(PseudoDreamEnv(CURRENT_ENV, render_mode="none"))
	eval_freq = 20_000
	n_eval_episodes = 5
	callback = EvalCallback(eval_env,
							eval_freq=eval_freq,
							n_eval_episodes=n_eval_episodes,
							verbose=1,
							best_model_save_path=CURRENT_ENV['data_dir'] + PPO_MODEL
						)
	total_steps = 500_000
	model.learn(total_timesteps=total_steps, progress_bar=True, callback=callback)
	final_mean, final_std = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
	print(f"Final evaluation: mean={final_mean:.2f} std={final_std:.2f}")
	model.save(CURRENT_ENV['data_dir'] + PPO_MODEL)