"""
orchestrator.py  —  runs on the MAC

Implements the distributed training loop:

  for each round:
    1. Gather real-robot data locally
    2. rsync data  →  server
    3. SSH → server: run train_server.py --round N --exp_id ID   (blocks)
    4. rsync models  ←  server
    5. Evaluate with updated models

The robot control / data collection still happens on the Mac because
it needs direct hardware access. All GPU training is delegated to the server.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import *

import time
import json
import torch
from stable_baselines3.ppo import PPO

from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from helpers.remote import rsync_push_data, rsync_pull_models, ssh_train_on_server
from envs.wrapper import evaluate_gathering_safe, generate_data
from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized

SMOOTHING = True if SMOOTH > 0 else False


def main():
    timings = dict(
        collecting_time        = 0.0,
        rsync_push_time        = 0.0,
        server_training_time   = 0.0,
        rsync_pull_time        = 0.0,
        evaluation_time        = 0.0,
    )

    start_time = time.time()

    # ------------------------------------------------------------------
    # Bootstrap: start with untrained (random) models for initial rollout.
    # After round 0 these will be replaced by the server's trained versions.
    # ------------------------------------------------------------------
    vq   = VQVAE(CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, 0.25, best_device(), True)
    lstm = LSTMQuantized(vq, best_device(), CURRENT_ENV['a_size'], PROP_SIZE, HIDDEN_DIM)

    with open(LOG_NAME + '.csv', 'w') as f:
        f.write('mrew,success,space,max_space,min_space,std\n')

    # ------------------------------------------------------------------
    # Initial data collection (no policy yet → random / open-loop)
    # ------------------------------------------------------------------
    print("\n[orchestrator] === Initial data collection ===")
    t = time.time()
    evaluate_gathering_safe(vq, lstm, policy=None, n_sample=200, training_set=True,  round=EXP_ID)
    evaluate_gathering_safe(vq, lstm, policy=None, n_sample=200, training_set=False, round=EXP_ID)
    timings['collecting_time'] += time.time() - t

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for round_idx in range(N_ROUNDS):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_idx + 1} / {N_ROUNDS}")
        print(f"{'='*60}")

        # ---- 1. Push data to server ----------------------------------
        print(f"\n[orchestrator] Step 1/4 — push data to server")
        t = time.time()
        rsync_push_data()
        timings['rsync_push_time'] += time.time() - t

        # ---- 2. Trigger training on server (blocking SSH call) -------
        print(f"\n[orchestrator] Step 2/4 — remote training on server")
        t = time.time()
        ssh_train_on_server(round_idx=round_idx, exp_id=EXP_ID)
        timings['server_training_time'] += time.time() - t

        # ---- 3. Pull trained models back to Mac ----------------------
        print(f"\n[orchestrator] Step 3/4 — pull models from server")
        t = time.time()
        rsync_pull_models()
        timings['rsync_pull_time'] += time.time() - t

        # Reload updated checkpoints so evaluate_gathering uses them
        vq   = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, best_device())
        lstm = load_lstm_quantized(CURRENT_ENV, vq, best_device(), HIDDEN_DIM, SMOOTHING, cl=False, kl=False)

        # ---- 4. Evaluate & collect new data with trained policy ------
        print(f"\n[orchestrator] Step 4/4 — evaluate + collect new data")
        t = time.time()

        # Load the PPO agent checkpoint that was saved by the server
        agent_path = CURRENT_ENV['models'] + 'agent' + f'{EXP_ID}'
        try:
            agent = PPO.load(agent_path)
        except FileNotFoundError:
            print("[orchestrator] WARNING: agent checkpoint not found, using None policy")
            agent = None

        rew, succ = evaluate_gathering_safe(vq, lstm, n_sample=100, policy=agent, training_set=True, round=EXP_ID)

        with open(LOG_NAME + '.csv', 'a') as f:
            for i in range(len(rew)):
                space     = torch.mean(torch.abs(vq.quantizer.embedding.weight.data))
                max_space = torch.max(vq.quantizer.embedding.weight.data)
                min_space = torch.min(vq.quantizer.embedding.weight.data)
                std_space = torch.std(vq.quantizer.embedding.weight.data)
                f.write(f'{rew[i]:.3f},{succ[i]},{space:.3f},{max_space:.3f},{min_space:.3f},{std_space:.5f}\n')

                if not torch.isfinite(space):
                    print("[orchestrator] NaN detected in codebook — stopping.")
                    _save_timings(timings, start_time)
                    sys.exit(1)

        timings['collecting_time'] += time.time() - t

        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"\n\033[1;31m--- {elapsed} ---\033[0m")

    _save_timings(timings, start_time)


def _save_timings(timings: dict, start_time: float) -> None:
    timings['total_time'] = time.time() - start_time
    with open(LOG_NAME + 'time.json', 'w') as f:
        json.dump(timings, f, indent=2)
    print("\n[orchestrator] Timing breakdown:")
    for k, v in timings.items():
        print(f"  {k:<26}: {time.strftime('%H:%M:%S', time.gmtime(v))}")


if __name__ == '__main__':
    main()
