# Visual PPO Baseline for Meta-World

This folder provides an optimized **visual Reinforcement Learning baseline** using **Proximal Policy Optimization (PPO)** on **Meta-World** robotic manipulation tasks (e.g. `button-press-topdown-v3`).

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Image Preprocessing & Frame Stacking](#image-preprocessing--frame-stacking)
3. [Impala CNN Backbone (Dreamer-style)](#impala-cnn-backbone-dreamer-style)
4. [Vision PPO Hyperparameters & Best Practices](#vision-ppo-hyperparameters--best-practices)
5. [Environment Randomness Mechanics](#environment-randomness-mechanics)
6. [Success Rate & Metrics Tracking](#success-rate--metrics-tracking)
7. [How to Train](#how-to-train)
8. [How to Monitor with TensorBoard](#how-to-monitor-with-tensorboard)
9. [How to Test, Evaluate & Record Video](#how-to-test-evaluate--record-video)
10. [Configuration Reference (`global_var.py`)](#configuration-reference-global_varpy)
11. [Next High-Impact Improvements to Try](#next-high-impact-improvements-to-try)

---

## Overview & Architecture

Learning robotic manipulation directly from raw pixels requires addressing:
- **Partial Observability**: A single still frame lacks velocity and movement direction.
- **High-Dimensional Latent Spaces**: Visual representations can overfit to textures and shadows.
- **Sample Inefficiency**: On-policy PPO discards rollouts after each update.

This implementation integrates modern image RL practices to maximize stability and sample efficiency:
- **Observation Space**: Sequence of 3 stacked black-and-white frames `(3, 64, 64)`.
- **Feature Extractor**: Residual **Impala CNN** (the same architecture used in *Dreamer* and *Procgen*).
- **Environment Vectorization**: 4 parallel environments with individual seed offsets.
- **Reward Normalization**: Running discounted return scaling (`VecNormalize`).
- **Learning Rate Annealing**: Linear decay from `3e-4` to `1e-5`.

---

## Image Preprocessing & Frame Stacking

Implemented in [`env_wrapper.py`](env_wrapper.py):

* **Grayscale Conversion**:
  - Raw RGB frames $(64, 64, 3)$ are converted to single-channel $(64, 64)$ via OpenCV (`cv2.COLOR_RGB2GRAY`).
  - Retains critical geometric edges and contact information while reducing channel footprint by $3\times$.
* **Frame Stacking (3 Frames)**:
  - Maintained using a rolling buffer (`collections.deque(maxlen=3)`).
  - On `reset()`, the buffer is seeded with 3 copies of the initial frame.
  - On `step()`, new frames are pushed while the oldest is discarded.
  - Resulting observation shape: **`(3, 64, 64)`**.
  - **Why**: Allows the CNN to infer **velocity, acceleration, and motion direction** of the robotic arm relative to the object (restoring the Markov property).
* **Channels-First (`C, H, W`) & Memory Efficiency (`uint8`)**:
  - Stored as `np.uint8` (`0` to `255`), which reduces rollout RAM by $4\times$.
  - Stable-Baselines3 automatically normalizes observations to `[0.0, 1.0]` on the compute device (`mps` / `cuda`).
* **Action Repeat (Frame Skip = 2)**:
  - Each action is repeated for 2 simulation steps, accumulating rewards.
  - Produces noticeable motion between consecutive frames and shortens the effective task horizon.

---

## Impala CNN Backbone (Dreamer-style)

Implemented in [`impala_cnn.py`](impala_cnn.py):

Default SB3 uses `NatureCNN` (3 simple Conv layers without residual connections). We replaced it with **Impala CNN** (*Espeholt et al., 2018*), standard in *Dreamer* (*Hafner et al.*):

```
Input (3, 64, 64)
  │
  ├── Stage 1: Conv(16) -> MaxPool(stride 2) -> ResBlock(16) -> ResBlock(16)   => (16, 32, 32)
  ├── Stage 2: Conv(32) -> MaxPool(stride 2) -> ResBlock(32) -> ResBlock(32)   => (32, 16, 16)
  └── Stage 3: Conv(32) -> MaxPool(stride 2) -> ResBlock(32) -> ResBlock(32)   => (32, 8, 8)
  │
  └── Flatten -> ReLU -> Linear(2048, 256) -> Output Latent Vector (256)
```

- **Residual Blocks**: Each block computes $x \leftarrow x + \text{Conv}(\text{ReLU}(\text{Conv}(\text{ReLU}(x))))$, preventing vanishing gradients in deeper visual networks.
- Total Parameters: ~622,000 (compact, expressive, and runs efficiently on Apple Silicon `mps` or NVIDIA `cuda`).

---

## Vision PPO Hyperparameters & Best Practices

| Hyperparameter | Value | Description / Rationale |
| :--- | :--- | :--- |
| `N_ENVS` | `4` | Number of parallel vectorized environments (`DummyVecEnv` on macOS) |
| `PPO_N_STEPS` | `1024` | Rollout length per env ($4 \times 1024 = 4096$ total steps per batch) |
| `PPO_BATCH_SIZE`| `128` | Mini-batch size (divides 4096 into 32 smooth gradient batches) |
| `PPO_N_EPOCHS` | `10` | Optimization passes over each rollout buffer |
| `PPO_LR` | `3e-4` | Initial learning rate |
| `PPO_MIN_LR` | `1e-5` | Final learning rate after linear annealing decay |
| `NORMALIZE_REWARD` | `True` | Scales returns with `VecNormalize` to stabilize value loss |
| `PPO_ENT_COEF` | `0.005` | Entropy coefficient to maintain continuous Gaussian action exploration |
| `clip_range` | `0.2` | PPO surrogate clipping threshold |
| `gamma` / `gae_lambda` | `0.99` / `0.95` | Discount factor and GAE smoothing factor |

---

## Environment Randomness Mechanics

Every time an episode resets (`env.reset()`):
- **What is Random**: The target object (e.g. the button box) **moves randomly** on the table within:
  - **X-axis (Left/Right)**: Uniform `[-0.10, +0.10]` m (**20 cm span**)
  - **Y-axis (Forward/Back)**: Uniform `[0.80, 0.90]` m (**10 cm span**)
  - **Z-axis (Height)**: Fixed at `0.115` m
- **What is Fixed**:
  - The Sawyer robotic arm always starts from the same home configuration above the table.
  - Camera viewpoint and angle remain static.
  - MuJoCo physics simulation is deterministic.

> **Tip for Debugging**: If you want to train or evaluate on a **fixed** button position first:
> ```python
> env.unwrapped._freeze_rand_vec = True
> ```

---

## Success Rate & Metrics Tracking

Meta-World tasks can succeed momentarily during an episode. The wrapper tracks success across the entire episode:
```python
if float(step_info.get("success", 0.0)) > 0.0:
    self.episode_success = 1.0

info["is_success"] = bool(self.episode_success > 0.0)
```
- Stable-Baselines3 registers `info["is_success"]` upon episode termination.
- Automatically computes and logs the rolling average under **`rollout/success_rate`** in both console and TensorBoard.

---

## How to Train

Run the training pipeline:
```bash
conda activate rl_env
python env_wrapper.py
```

Upon completion, it saves:
- `ppo_metaworld_vision.zip`: Trained policy weights.
- `vec_normalize.pkl`: Running reward normalization statistics (essential for evaluation).

---

## How to Monitor with TensorBoard

Launch the TensorBoard server:
```bash
conda activate rl_env
tensorboard --logdir ./tensorboard_logs/
```
Open **`http://localhost:6006`** in your browser.

### Key Metrics
- **`rollout/success_rate`**: Percentage of successful episodes ($0.0$ to $1.0$).
- **`rollout/ep_rew_mean`**: Mean episode reward.
- **`rollout/ep_len_mean`**: Mean episode duration.
- **`train/entropy_loss`**: Exploration entropy (should decline smoothly, not instantly collapse).
- **`train/value_loss`**: Critic prediction error.

### TensorBoard Tips
- **Auto-Reload**: Click the **Gear icon (⚙️)** in the top right $\rightarrow$ check **Reload data** $\rightarrow$ set to 15s.
- **Instant Refresh**: Press **`r`** anywhere on the webpage.

---

## How to Test, Evaluate & Record Video

Run [`test_ppo.py`](test_ppo.py) to evaluate a trained model:

```bash
# 1. Standard evaluation (10 episodes with MP4 recording)
python test_ppo.py

# 2. Evaluate a specific checkpoint for 20 episodes
python test_ppo.py --model-path ppo_metaworld_vision.zip --n-episodes 20

# 3. Test on a fixed (frozen) button position
python test_ppo.py --freeze-target --n-episodes 10

# 4. Fast evaluation without video recording
python test_ppo.py --no-video --n-episodes 50
```

---

## Configuration Reference (`global_var.py`)

All core parameters are centralized in [`global_var.py`](global_var.py):

```python
CURRENT_ENV = BUTTON_TD          # Target task config (e.g. BUTTON_TD, PUSH, PEG_INSERT)
N_ENVS = 4                       # Parallel environments
PPO_N_STEPS = 1024               # Rollout steps per environment
PPO_BATCH_SIZE = 128             # PPO mini-batch size
PPO_N_EPOCHS = 10                # Optimization epochs per rollout
PPO_LR = 0.0003                  # Initial learning rate
PPO_MIN_LR = 1e-5                # Final learning rate after decay
PPO_ENT_COEF = 0.005             # Exploration entropy bonus
NORMALIZE_REWARD = True          # Enable VecNormalize return scaling
USE_IMPALA = True                # True = ImpalaCNN, False = NatureCNN
PPO_FEATURES_DIM = 256           # Latent feature dimension
IMPALA_DEPTHS = (16, 32, 32)     # Channel depths for Impala stages
FRAME_STACK = 3                  # Number of consecutive frames
GRAYSCALE = True                 # Convert RGB to single-channel grayscale
CHANNELS_FIRST = True            # (C, H, W) PyTorch format
ACTION_REPEAT = True             # Action repeat / frame skip
ACTION_REPEAT_STEPS = 2          # Steps repeated per action
```

---

## Next High-Impact Improvements to Try

If you want to push success rates even higher:

1. **Multi-Modal State (Vision + Proprioception)**:
   - Combine the stacked camera images with the 4D robot end-effector coordinates `(x, y, z, gripper_open)`.
   - Uses SB3's `MultiInputPolicy`, relieving the CNN from having to infer the robot arm's own 3D position purely from pixels.
2. **Visual Data Augmentation (DrQ / RAD style)**:
   - Apply random image padding and cropping ($\pm 4$ pixels) during training to prevent the CNN from memorizing static visual artifacts.
3. **Off-Policy Algorithms (DrQ-v2 / SAC)**:
   - Off-policy methods replay past visual transitions from a buffer, typically reaching $\ge 90\%$ success rates in 100k–300k steps compared to several million steps for PPO.
