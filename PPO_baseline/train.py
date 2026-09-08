import os
import sys
import csv
import platform
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

# Adjust paths to match your project structure
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers import best_device
from global_var import *
from impala_cnn import ImpalaCNN
from env_wrapper import make_vec_envs, linear_schedule


class CsvEvalCallback(BaseCallback):
    """
    Custom callback for evaluating the agent every `eval_freq` steps.
    Runs `n_eval_episodes` concurrently in a vectorized environment and saves results to a CSV.
    """
    def __init__(self, eval_env, eval_freq: int, csv_path: str = 'eval_results.csv', verbose: int = 1):
        super(CsvEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_envs = eval_env.num_envs
        self.csv_path = csv_path
        self.next_eval_step = self.eval_freq

        # Initialize CSV and write header if it doesn't exist
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['training_step', 'eval_run', 'mrew', 'success'])

    def _on_step(self) -> bool:
        # Check if we reached the evaluation frequency
        if self.num_timesteps >= self.next_eval_step:
            self._run_evaluation()
            self.next_eval_step += self.eval_freq
        return True

    def _run_evaluation(self):
        if self.verbose > 0:
            print(f"\n[Eval] Triggering parallel evaluation for {self.n_envs} episodes at {self.num_timesteps} steps...")
        
        obs = self.eval_env.reset()
        
        # Track state for each parallel environment
        ep_rewards = np.zeros(self.n_envs)
        ep_successes = np.zeros(self.n_envs, dtype=bool)
        ep_dones = np.zeros(self.n_envs, dtype=bool)
        
        # Step until ALL environments have completed one episode
        while not np.all(ep_dones):
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            
            for i in range(self.n_envs):
                # Only add rewards/success if the environment hasn't finished its first episode yet
                # (VecEnv auto-resets, so we must ignore steps after the first 'done')
                if not ep_dones[i]:
                    ep_rewards[i] += rewards[i]
                    
                    if infos[i].get("is_success", False):
                        ep_successes[i] = True
                        
                    if dones[i]:
                        ep_dones[i] = True
                        
        # Format results for CSV
        results = []
        for i in range(self.n_envs):
            results.append([self.num_timesteps, i + 1, ep_rewards[i], ep_successes[i]])
            
        # Append results to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
            
        # Compute means for logging
        mean_reward = np.mean(ep_rewards)
        mean_success = np.mean(ep_successes)
        
        if self.verbose > 0:
            print(f"[Eval Results] Mean Reward: {mean_reward:.2f} | Mean Success Rate: {mean_success * 100:.1f}%\n")


if __name__ == "__main__":
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument("--run-name", type=str, default="default", help="Unique name for this run (used for output files)")
    parser.add_argument("--n-eval", type=int, default=10, help="Number of evaluation episodes to run concurrently")
    args = parser.parse_args()

    # Dynamic filenames based on the run name
    csv_filename = f"evaluation_metrics_{args.run_name}.csv"
    model_filename = f"ppo_metaworld_vision_{args.run_name}"
    vecnorm_filename = f"vec_normalize_{args.run_name}.pkl"

    # Setup rendering vars
    if 'MUJOCO_GL' not in os.environ:
        if platform.system() == 'Darwin':  # macOS
            os.environ['MUJOCO_GL'] = 'glfw'
        else:  # Linux / servers
            os.environ['MUJOCO_GL'] = 'egl'

    device = best_device()
    print(f"Device: {device} | Run Name: {args.run_name}")
    
    # 1. Initialize training environment
    use_subproc = (platform.system() != 'Darwin')
    train_env = make_vec_envs(
        env_dict=CURRENT_ENV,
        n_envs=N_ENVS,
        frame_stack=FRAME_STACK,
        grayscale=GRAYSCALE,
        channels_first=CHANNELS_FIRST,
        action_repeat=(ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
        normalize_reward=NORMALIZE_REWARD,
        use_subproc=use_subproc,
    )
    
    # 2. Initialize a separate PARALLEL evaluation environment
    # We set n_envs = args.n_eval so it runs all evaluation episodes at the exact same time
    eval_env = make_vec_envs(
        env_dict=CURRENT_ENV,
        n_envs=args.n_eval, 
        frame_stack=FRAME_STACK,
        grayscale=GRAYSCALE,
        channels_first=CHANNELS_FIRST,
        action_repeat=(ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
        normalize_reward=False,  
        use_subproc=use_subproc, # Enabled subproc here to parallelize eval!
    )

    # 3. Configure architecture
    policy_kwargs = {}
    if USE_IMPALA:
        print(f"Backbone: ImpalaCNN (residual blocks, depths={IMPALA_DEPTHS}, dim={PPO_FEATURES_DIM})")
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=PPO_FEATURES_DIM,
                depths=IMPALA_DEPTHS,
            ),
        )
    else:
        print("Backbone: NatureCNN (default SB3)")

    # 4. Instantiate PPO
    lr_schedule = linear_schedule(PPO_LR, final_value=PPO_MIN_LR)
    agent = PPO(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=PPO_ENT_COEF,
        verbose=0,
        device=device,
        tensorboard_log=f"./tensorboard_logs/{args.run_name}/",
    )

    # 5. Calculate frequencies and setup the callback
    steps_per_episode = 500 if not ACTION_REPEAT else 250
    eval_freq = 10 * steps_per_episode
    total_steps = N_ROUNDS * steps_per_episode
    
    print(f"Starting PPO training for {total_steps} timesteps...")
    print(f"Evaluation scheduled every {eval_freq} steps ({args.n_eval} parallel episodes).")
    
    eval_callback = CsvEvalCallback(
        eval_env=eval_env, 
        eval_freq=eval_freq, 
        csv_path=csv_filename
    )

    # 6. Train the agent
    agent.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    # 7. Save models and normalize variables
    agent.save(model_filename)
    print(f"Model saved to {model_filename}.zip")
    
    if NORMALIZE_REWARD and isinstance(train_env, VecNormalize):
        train_env.save(vecnorm_filename)
        print(f"VecNormalize statistics saved to {vecnorm_filename}")

    train_env.close()
    eval_env.close()