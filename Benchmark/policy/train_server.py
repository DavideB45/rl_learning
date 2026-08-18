"""
train_server.py  —  runs on the LINUX training server

Called remotely by the Mac via SSH:
    python train_server.py --round <N> --exp_id <ID>

It expects data to already be present (pushed by the Mac via rsync before
the SSH call). After training it saves checkpoints to the models directory,
which the Mac will pull back via rsync.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import *

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = GPU_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import argparse
import numpy as np
from torch.optim import Adam
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO
from torch.utils.data import DataLoader

from helpers.data import make_image_dataloader_safe, make_seq_dataloader_safe, get_data_path
from helpers.model_loader import (
    load_vq_vae, load_lstm_quantized,
    save_vq_vae, save_lstm_quantized,
)
from helpers.general import best_device
from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from envs.simulator import SoftDreamEnv

SMOOTHING = True if SMOOTH > 0 else False
BASE = 'home/davide/github/rl_learning/Benchmark/'
policy_kwargs = dict(
    net_arch=dict(
        pi=[1024, 512, 256],
        vf=[1024, 512, 256],
    ),
    ortho_init=True,
)

colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
reset  = '\033[0m'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Server-side world-model training for one round.")
    p.add_argument("--round",  type=int, required=True,  help="Current training round index (0-based)")
    p.add_argument("--exp_id", type=str, required=True,  help="Experiment ID (used for file paths)")
    return p.parse_args()


def main():
    args = parse_args()
    round_idx = args.round
    exp_id    = args.exp_id

    print(f"\n{'='*60}")
    print(f"  SERVER TRAINING  |  round={round_idx}  exp={exp_id}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load or initialise models
    # ------------------------------------------------------------------
    # Round 0: build from scratch.
    # Later rounds: load the checkpoint saved at the end of the previous round
    # so we do incremental fine-tuning, not re-training from random weights.
    if round_idx == 0:
        vq   = VQVAE(CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, 0.25, best_device(), True)
        lstm = LSTMQuantized(vq, best_device(), CURRENT_ENV['a_size'], PROP_SIZE, HIDDEN_DIM)
        agent = None
    else:
        print("[server] Loading previous checkpoints …")
        vq    = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, best_device())
        lstm  = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTHING, cl=False, kl=False)
        agent = None  # PPO is loaded inside tune_agent when path exists

    # ------------------------------------------------------------------
    # VQ-VAE
    # ------------------------------------------------------------------
    vq_epochs = VQ_EPOCS if round_idx == 0 else 1
    vq = tune_vq(model=vq, num_epocs = vq_epochs, lr = VQ_LR / np.log(round_idx * 5 + 4), reg = SMOOTH, wd = VQ_WD, exp_id = exp_id)

    # ------------------------------------------------------------------
    # Sequence dataloaders (built after VQ is updated so codes are fresh)
    # ------------------------------------------------------------------
    tr_seq = make_seq_dataloader_safe( get_data_path(BASE + CURRENT_ENV['img_dir'], True,  exp_id), vq, SEQ_LEN, 128, max_ep = EP_ON_LOOP )
    vl_seq = make_seq_dataloader_safe( get_data_path(BASE + CURRENT_ENV['img_dir'], False, exp_id), vq, SEQ_LEN, 128, max_ep = 15, )

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------
    lstm_epochs = LSTM_EPOCS if round_idx == 0 else 1
    lstm = tune_lstm( model = lstm, tr = tr_seq, vl = vl_seq, encoder = vq, num_epocs = lstm_epochs, lr = LSTM_LR, wd = LSTM_WD)

    # ------------------------------------------------------------------
    # PPO agent in dream environment
    # ------------------------------------------------------------------
    dream_env = SoftDreamEnv( vq, lstm, vl_seq, init_len = INIT_LEN, ep_len = DREAM_LEN, num_envs = 50 )
    agent = tune_agent(agent, num_steps=PPO_STEPS, env=dream_env)

    print(f"\n[server] Round {round_idx} complete. Checkpoints saved.\n")


# ---------------------------------------------------------------------------
# Training subroutines (same logic as original main.py)
# ---------------------------------------------------------------------------

def tune_vq(model: VQVAE, num_epocs: int, lr: float, wd: float, reg: float, exp_id: str) -> VQVAE:
    tr = make_image_dataloader_safe(get_data_path(BASE + CURRENT_ENV['img_dir'], True,  exp_id), max_size=EP_ON_LOOP * 500)
    vl = make_image_dataloader_safe(get_data_path(BASE + CURRENT_ENV['img_dir'], False, exp_id), max_size=1500)
    optim = Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_loss  = float('inf')
    no_improvements = 0

    for epoch in range(num_epocs):
        print("-" * 25 + f" VQ {(epoch + 1):02}/{num_epocs} " + "-" * 25)
        tr_loss  = model.train_epoch(tr,  optim, reg)
        val_loss = model.eval_epoch(vl,   reg)

        if val_loss['total_loss'] < best_val_loss:
            best_val_loss = val_loss['total_loss']
            save_vq_vae(CURRENT_ENV, model, smooth=SMOOTHING)
            print(f"{colors[-1]}  New best VQ-VAE saved!{reset}")
        else:
            no_improvements += 1
            if no_improvements >= 3:
                print("[server] VQ early stop.")
                break

        for i, key in enumerate(tr_loss):
            color = colors[i % len(colors)]
            print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")

    del model
    return load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, best_device())


def tune_lstm(model: LSTMQuantized, tr: DataLoader, vl: DataLoader,
              encoder: VQVAE, num_epocs: int, lr: float, wd: float) -> LSTMQuantized:
    model.quantizer = encoder
    optim = Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_loss   = float('inf')
    no_improvements = 0

    for epoch in range(num_epocs):
        err_tr = model.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, rew_weight=REW_WEIGHT)
        err_vl = model.eval_rwm_style(vl,         init_len=INIT_LEN, err_decay=0.99, rew_weight=REW_WEIGHT)

        if err_vl['mse'] < best_val_loss:
            print_lstm_analytics(epoch, err_tr, err_vl)
            best_val_loss   = err_vl['mse']
            no_improvements = 0
            save_lstm_quantized(CURRENT_ENV, model, cl=False, kl=False, tf=SMOOTHING)
        else:
            no_improvements += 1
            if no_improvements >= 5:
                print("[server] LSTM early stop.")
                break

    if num_epocs == 1:
        return model

    del model
    model = load_lstm_quantized(CURRENT_ENV, encoder, best_device(), HIDDEN_DIM, SMOOTHING, cl=False, kl=False)
    model.compile()
    return model


def tune_agent(agent: PPO, env: SoftDreamEnv, num_steps: int) -> PPO:
    agent_path = CURRENT_ENV['models'] + 'agent' + f'{EXP_ID}'
    if agent is None:
        try:
            agent = PPO.load(agent_path, env)
            print("[server] Loaded existing PPO agent checkpoint.")
        except FileNotFoundError:
            print("[server] No existing agent found — initialising fresh PPO.")
            agent = PPO( MlpPolicy, env, policy_kwargs = policy_kwargs, n_steps = 500, batch_size = 1000, learning_rate = PPO_LR, ent_coef = 0.01, sde_sample_freq = 10, use_sde = True )
    agent = agent.learn(num_steps, progress_bar=True, reset_num_timesteps=False)
    agent.save(agent_path)
    return agent


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

PURPLE = "\033[95m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; RESET = "\033[0m"
COL1, COL2, COL3 = 15, 12, 12
WIDTH = COL1 + COL2 + COL3 + 6

def row(c1, c2="", c3="", color=RESET):
    print(color + f"| {c1:<{COL1}} | {c2:>{COL2}} | {c3:>{COL3}} |" + RESET)

def sep(color=RESET):
    print(color + "+" + "-" * (WIDTH + 2) + "+" + RESET)

def print_lstm_analytics(epoch, err_tr, err_vl):
    sep(PURPLE)
    row(f"Epoch {epoch}", "Train", "Val", YELLOW)
    sep(PURPLE)
    row("MSE",       f"{err_tr['mse']:.4f}",       f"{err_vl['mse']:.4f}",       BLUE)
    row("QMSE",      f"{err_tr['qmse']:.4f}",      f"{err_vl['qmse']:.4f}",      BLUE)
    row("Prop MSE",  f"{err_tr['prop_mse']:.4f}",  f"{err_vl['prop_mse']:.4f}",  BLUE)
    row("Rew MSE",   f"{err_tr['reward_mse']:.4f}",f"{err_vl['reward_mse']:.4f}",BLUE)
    row("Accuracy",  f"{err_tr['acc']:.1f}%",      f"{err_vl['acc']:.1f}%",      PURPLE)
    row("First Acc", f"{err_tr['first_acc']:.1f}%",f"{err_vl['first_acc']:.1f}%",PURPLE)
    sep(PURPLE)


if __name__ == '__main__':
    main()
