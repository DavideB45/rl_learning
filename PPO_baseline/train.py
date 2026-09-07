import os
import sys
import csv
import platform
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
    Runs `n_eval_episodes` episodes and saves reward and success of EACH run to a CSV.
    """
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 10, csv_path: str = 'eval_results.csv', verbose: int = 1):
        super(CsvEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
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
            print(f"\n[Eval] Triggering evaluation at {self.num_timesteps} steps...")
        
        results = []
        
        for ep in range(1, self.n_eval_episodes + 1):
            # VecEnv returns only the observation on reset
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_success = False
            
            while not done:
                # Predict action deterministically for evaluation
                action, _ = self.model.predict(obs, deterministic=True)
                
                # VecEnv returns exactly 4 elements
                obs, reward, dones, infos = self.eval_env.step(action)
                
                ep_reward += reward[0]
                
                # Check for success metric stored by your wrapper
                if infos[0].get("is_success", False):
                    ep_success = True
                    
                done = dones[0]
                
            results.append([self.num_timesteps, ep, ep_reward, ep_success])
            
        # Append results to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
            
        # Compute means for logging
        mean_reward = np.mean([r[2] for r in results])
        mean_success = np.mean([r[3] for r in results])
        
        if self.verbose > 0:
            print(f"[Eval Results] Mean Reward: {mean_reward:.2f} | Mean Success Rate: {mean_success * 100:.1f}%\n")


if __name__ == "__main__":
    # Setup rendering vars
    if 'MUJOCO_GL' not in os.environ:
        if platform.system() == 'Darwin':  # macOS
            os.environ['MUJOCO_GL'] = 'glfw'
        else:  # Linux / servers
            os.environ['MUJOCO_GL'] = 'egl'

    device = best_device()
    print(f"Device: {device}")
    
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
    
    # 2. Initialize a separate SINGLE evaluation environment
    # Note: normalize_reward is False so evaluation is on raw true rewards!
    eval_env = make_vec_envs(
        env_dict=CURRENT_ENV,
        n_envs=1, 
        frame_stack=FRAME_STACK,
        grayscale=GRAYSCALE,
        channels_first=CHANNELS_FIRST,
        action_repeat=(ACTION_REPEAT_STEPS if ACTION_REPEAT else 1),
        normalize_reward=False,  
        use_subproc=False,
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
        verbose=1,
        device=device,
        tensorboard_log="./tensorboard_logs/",
    )

    # 5. Calculate frequencies and setup the callback
    steps_per_episode = 500 if not ACTION_REPEAT else 250
    eval_freq = 10 * steps_per_episode
    total_steps = N_ROUNDS * steps_per_episode
    
    print(f"Starting PPO training for {total_steps} timesteps...")
    print(f"Evaluation scheduled every {eval_freq} steps (10 episodes).")
    
    eval_callback = CsvEvalCallback(
        eval_env=eval_env, 
        eval_freq=eval_freq, 
        n_eval_episodes=10, 
        csv_path="evaluation_metrics.csv"
    )

    # 6. Train the agent
    agent.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    # 7. Save models and normalize variables
    agent.save("ppo_metaworld_vision")
    print("Model saved to ppo_metaworld_vision.zip")
    
    if NORMALIZE_REWARD and isinstance(train_env, VecNormalize):
        train_env.save("vec_normalize.pkl")
        print("VecNormalize statistics saved to vec_normalize.pkl")

    train_env.close()
    eval_env.close()