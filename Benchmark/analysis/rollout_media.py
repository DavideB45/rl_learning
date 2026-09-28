"""
Generates thesis-ready media for a single trained checkpoint:
  1. the real rollout, native 64x64 frames upsampled to UPSCALE_SIZE keeping the pixelated look
  2. the same starting setup and the same actions, but with transitions predicted by the
     world model (VQ-VAE + LSTM) instead of the real simulator, from the same context frames
  3. the same episode (same seed, same actions) re-rendered natively at high resolution
  4. a paper-ready figure: sampled frames from the real rollout with the imagined
     counterpart stacked below them

Only Meta-World simulated tasks are supported: the high-resolution re-render works by
replaying the recorded actions in a freshly-constructed env at a larger render size, which
needs a simulator (the real-robot "real-soft" checkpoint has none).
"""
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import imageio
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../'))

import gymnasium as gym
import metaworld  # noqa: F401 - registers Meta-World/MT1 with gymnasium
from stable_baselines3.ppo import PPO

from helpers.general import best_device
import envs.wrapper as wrapper_module
from helpers.model_loader import load_vq_vae
from envs.wrapper import MetaWrapEnv
from dynamics.lstm import LSTMQuantized
from global_var import (
	WINDOW_OPEN, DRAWER_OPEN, BUTTON_TD, PEG_INSERT,
	WINDOWO_DATA_DIR, DRAWERO_DATA_DIR, BUTTON_TD_DATA_DIR, PEG_DATA_DIT,
	CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, HIDDEN_DIM, SMOOTH,
	INIT_LEN, ACTION_REPEAT,
)

# ==========================================
# CONFIGURATION - pick the trained checkpoint to visualize.
# ==========================================
ENV_NAME = "peg-insert"  # one of: "window-open", "drawer-open", "button-press-td"
_ENV_CONFIGS = {"window-open": WINDOW_OPEN, "drawer-open": DRAWER_OPEN, "button-press-td": BUTTON_TD, "peg-insert": PEG_INSERT}
_ENV_ROOTS = {"window-open": WINDOWO_DATA_DIR, "drawer-open": DRAWERO_DATA_DIR, "button-press-td": BUTTON_TD_DATA_DIR, "peg-insert": PEG_DATA_DIT}
# These three checkpoints all predate the global_var.py change that decoupled PROP_SIZE from
# a_size (bumped to 8 for the later real-robot experiment); their LSTM was trained with
# prop_dim == a_size == 4, so PROP_SIZE from global_var can't be used to load them.
_ENV_PROP_SIZE = {"window-open": 4, "drawer-open": 4, "button-press-td": 4, "peg-insert": 4}

SEED = 11             # fixes the object/goal placement replayed across all outputs
MAX_STEPS = 50       # safety cap on wrapper-steps (episode usually ends earlier)
UPSCALE_SIZE = 512    # pixelated upsample target for videos 1 & 2
HQ_SIZE = 512         # native (non-pixelated) render size for video 3
FPS = 10

# paper figure options
FIGURE_STEP = 5       # steps between consecutive sampled frames (e.g. 10 -> every 10th step);
                          # takes priority over N_FIGURE_COLUMNS below when set
N_FIGURE_COLUMNS = 6      # used only when FIGURE_STEP is None: this many evenly-spaced frames
FIGURE_INCLUDE_HQ = True  # also stack a high-quality row on top of the ground truth/world model rows

class LegacyLSTMQuantized(LSTMQuantized):
	"""
	dynamics/lstm.py's merge step used to concatenate only [rep, action] (2*hidden_dim)
	into merge_fc; proprioception was run through pro_fc but never actually concatenated
	in (a bug in that era, fixed later when merge_fc was widened to 3*hidden_dim). The
	button-press-td/drawer-open/window-open checkpoints were all saved before that fix, so
	they need this narrower merge_fc and the old forward() to load and run correctly.
	"""

	def __init__(self, quantizer, device, action_dim, prop_dim, hidden_dim=512):
		super().__init__(quantizer, device, action_dim, prop_dim, hidden_dim)
		self.merge_fc = nn.Sequential(
			nn.LayerNorm(hidden_dim * 2),
			nn.Linear(hidden_dim * 2, hidden_dim * 2),
			nn.LeakyReLU(),
			nn.LayerNorm(hidden_dim * 2),
			nn.Linear(hidden_dim * 2, hidden_dim),
			nn.LeakyReLU(),
		).to(device)

	def forward(self, input, action, prop, h=None):
		input = self.flatten_rep(input)
		new_rep = self.rep_fc(input)
		action = self.act_fc(action)
		self.pro_fc(prop)  # computed but intentionally unused, matching the original checkpoint
		merged = torch.cat([new_rep, action], dim=-1)
		skip_output = self.merge_fc(merged)

		if h is None:
			h = (torch.zeros(1, input.size(0), self.hidden_dim).to(input.device),
				 torch.zeros(1, input.size(0), self.hidden_dim).to(input.device))
		output, h = self.lstm(skip_output, h)
		output = output + skip_output
		latent = self.out_fc(output)
		latent = self.unflatten_rep(latent, input.size(1))
		prop_out = self.out_prop_fc(output)
		reward = self.out_reward(output)

		latent_q = self.quantizer.quantizer.quantize_fixed_space(latent.reshape(-1, self.d, self.w_h, self.w_h))
		latent_q = latent_q.view(input.size(0), input.size(1), self.d, self.w_h, self.w_h)
		return latent, latent_q, prop_out, reward, h


class LegacyMetaWrapEnv(MetaWrapEnv):
	"""
	Before proprioception was appended to the observation vector, MetaWrapEnv's
	representation was just [latent, hidden_state] (no prop tail). The PPO agents for
	these three checkpoints were trained on that shorter observation, so this drops the
	prop tail the current MetaWrapEnv.reset()/step() append, to match what they expect.
	"""

	def reset(self, seed=None, options=None):
		obs, info = super().reset(seed=seed, options=options)
		return obs[:-self.dyn.prop_dim], info

	def step(self, action):
		obs, reward, terminated, truncated, info = super().step(action)
		return obs[:-self.dyn.prop_dim], reward, terminated, truncated, info


def load_legacy_lstm(env_cfg, vq, device, hidden_dim, prop_dim, smoothing):
	model = LegacyLSTMQuantized(vq, device, env_cfg['a_size'], prop_dim, hidden_dim)
	model_path = env_cfg['models'] + f"lstmq_{hidden_dim}_{vq.latent_dim}_{vq.code_depth}_{vq.codebook_size}"
	if smoothing:
		model_path += '_tf'
	model_path += ".pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

OUT_DIR = "images"


# ==========================================
# CHECKPOINT DISCOVERY
# ==========================================
def resolve_models_dir(env_root: str) -> str:
	"""
	Finds the checkpoint folder under env_root/models/*, without relying on EXP_ID from
	global_var.py: checkpoints across tasks were saved under folders whose numeric name
	doesn't always match the id baked into the agent's filename, so we just look for the
	VQ-VAE checkpoint that is always named the same way.
	"""
	for candidate in sorted(glob.glob(os.path.join(env_root, "models", "*"))):
		if glob.glob(os.path.join(candidate, "vq_*.pth")):
			return candidate + "/"
	raise FileNotFoundError(f"No VQ-VAE checkpoint found under {env_root}models/*")


def find_agent_checkpoint(models_dir: str) -> str:
	candidates = (sorted(glob.glob(os.path.join(models_dir, "agent_best*.zip")))
				  or sorted(glob.glob(os.path.join(models_dir, "agent*.zip"))))
	if not candidates:
		raise FileNotFoundError(f"No PPO agent checkpoint found in {models_dir}")
	return candidates[0][:-len(".zip")]


# ==========================================
# ROLLOUTS
# ==========================================
@torch.no_grad()
def rollout_real(env_cfg, vq, lstm, agent, seed, max_steps):
	"""Runs the policy on the real simulator, recording frames, latents, proprioception
	and actions so the same episode can be replayed by the world model and at high res."""
	env = LegacyMetaWrapEnv(vq, lstm, env_cfg=env_cfg, seed=seed)
	obs, _ = env.reset()
	# Meta-World's camera renders upside down, so every frame is flipped here once, right
	# at capture time, matching the .rotate(180) already used elsewhere in this codebase
	# (e.g. envs/wrapper.py's __main__) when displaying rendered frames.
	frames = [env.current_render.rotate(180)]
	latents = [env.current_latent.cpu()]
	props = [env.current_prop.cpu()]
	actions = []
	success = False
	for _ in range(max_steps):
		action, _ = agent.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		frames.append(env.current_render.rotate(180))
		latents.append(env.current_latent.cpu())
		props.append(env.current_prop.cpu())
		actions.append(action)
		success = success or (info.get("success") == 1)
		if terminated or truncated:
			break
	env.close()
	return frames, latents, props, actions, success


@torch.no_grad()
def imagine_rollout(vq, lstm, latents, props, actions, init_len, device):
	"""Feeds the first init_len real frames as context, then lets the LSTM predict the
	rest of the episode autoregressively using the actions actually taken."""
	T = len(actions)
	init_len = min(init_len, T - 1)
	if init_len <= 0:
		raise RuntimeError(f"Episode too short ({T} steps) to seed the world model with INIT_LEN={init_len}.")

	latent_seq = torch.cat([l.unsqueeze(1) for l in latents], dim=1).to(device)  # (1, T+1, D, W, H)
	prop_seq = torch.cat(props, dim=1).to(device)                               # (1, T+1, P)
	action_seq = torch.tensor(np.stack(actions), dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, A)

	lstm.eval()
	_, _, _, _, h = lstm.forward(latent_seq[:, :init_len], action_seq[:, :init_len], prop_seq[:, :init_len])
	_, gen_q, _, _, _ = lstm.ar_forward(
		latent_seq[:, init_len:init_len + 1], action_seq[:, init_len:], prop_seq[:, init_len:init_len + 1], h
	)
	imagined_imgs = vq.decode(gen_q.squeeze(0)).cpu()  # (T - init_len, 3, 64, 64) in [0, 1]
	return imagined_imgs, init_len


def render_hq_rollout(env_cfg, seed, actions, size):
	"""Replays the exact same episode (same construction seed, same actions) in a fresh
	env built at `size` resolution, giving a natively-rendered (non-pixelated) video."""
	env = gym.make('Meta-World/MT1', env_name=env_cfg['env_name'], render_mode='rgb_array',
					camera_id=env_cfg['camera_id'], width=size, height=size, seed=seed)
	env.env.env.env.env.env.env.env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
	env.reset()
	# same upside-down camera as rollout_real, flipped the same way at capture time
	frames = [Image.fromarray(env.render()).rotate(180)]
	for action in actions:
		env.step(action)
		if ACTION_REPEAT:
			env.step(action)
		frames.append(Image.fromarray(env.render()).rotate(180))
	env.close()
	return frames


# ==========================================
# MEDIA HELPERS
# ==========================================
def pil_pixelated(frame, size) -> Image.Image:
	if not isinstance(frame, Image.Image):
		frame = Image.fromarray(np.asarray(frame))
	return frame.convert("RGB").resize((size, size), Image.NEAREST)


def pil_smooth(frame, size) -> Image.Image:
	"""Like pil_pixelated but for already high-resolution frames: smooth resampling
	instead of nearest-neighbor, since there's no native pixel grid to preserve."""
	if not isinstance(frame, Image.Image):
		frame = Image.fromarray(np.asarray(frame))
	return frame.convert("RGB").resize((size, size), Image.LANCZOS)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
	arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
	# the VQ-VAE was trained on the same upside-down camera frames, so its decoded output
	# needs the same flip as rollout_real/render_hq_rollout to line up with them.
	return Image.fromarray(arr).rotate(180)


def save_video(frames, path, fps=FPS):
	imageio.mimsave(path, [np.array(f) for f in frames], fps=fps)
	print(f"Saved {path}")


def make_paper_figure(real_frames, imagined_imgs, init_len, out_path, env_label,
					   n_columns=6, step=None, hq_frames=None, cell_size=512):
	"""
	step: distance (in rollout steps) between consecutive sampled columns. Takes priority
		over n_columns when given (e.g. step=10 samples every 10th available step).
	n_columns: used only when step is None: this many evenly-spaced columns are sampled.
	hq_frames: when given, adds a "High quality" row on top using the same absolute step
		indices (must be aligned with real_frames, i.e. hq_frames[t] <-> real_frames[t]).
	"""
	n_available = imagined_imgs.shape[0]
	if n_available == 0:
		print("No imagined frames available, skipping paper figure.")
		return
	if step is not None:
		idx = list(range(0, n_available, step))
	else:
		n_columns = min(n_columns, n_available)
		idx = sorted(set(np.linspace(0, n_available - 1, n_columns).round().astype(int).tolist()))
	n_columns = len(idx)

	row_defs = []
	if hq_frames is not None:
		row_defs.append(("High quality", "#4e9a51"))
	row_defs.extend([("Ground truth", "#4e75a4"), ("World model", "#e34948")])
	n_rows = len(row_defs)

	fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.2 * n_columns, 2.3 * n_rows))
	if n_columns == 1:
		axes = axes[:, None]

	for col, i in enumerate(idx):
		t = init_len + 1 + i  # absolute rollout step this column corresponds to
		images = []
		if hq_frames is not None:
			images.append(pil_smooth(hq_frames[t], cell_size))
		images.append(pil_pixelated(real_frames[t], cell_size))
		images.append(pil_pixelated(tensor_to_pil(imagined_imgs[i]), cell_size))

		for row, (img, (label, color)) in enumerate(zip(images, row_defs)):
			ax = axes[row, col]
			ax.imshow(img, interpolation="nearest")
			ax.set_xticks([])
			ax.set_yticks([])
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_color(color)
				spine.set_linewidth(2)
			if row == 0:
				ax.set_title(f"t = {t}", fontsize=11)
			if col == 0:
				ax.set_ylabel(label, fontsize=12, fontweight="bold", color=color)

	fig.suptitle(f"Real vs. imagined rollout — {env_label}", fontsize=14, fontweight="bold")
	plt.tight_layout()
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {out_path}")


# ==========================================
# MAIN
# ==========================================
def main():
	os.makedirs(OUT_DIR, exist_ok=True)
	device = best_device()
	smoothing = True if SMOOTH else False

	env_cfg = dict(_ENV_CONFIGS[ENV_NAME])
	env_cfg["models"] = resolve_models_dir(_ENV_ROOTS[ENV_NAME])
	print(f"Using checkpoint at {env_cfg['models']}")

	vq = load_vq_vae(env_cfg, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, smoothing, device)
	# these checkpoints were trained before PROP_SIZE was decoupled from a_size (see
	# _ENV_PROP_SIZE above); patch wrapper.py's copy of the global so MetaWrapEnv slices the
	# proprioception vector to the size this checkpoint's LSTM actually expects.
	wrapper_module.PROP_SIZE = _ENV_PROP_SIZE[ENV_NAME]
	lstm = load_legacy_lstm(env_cfg, vq, device, HIDDEN_DIM, _ENV_PROP_SIZE[ENV_NAME], smoothing)
	agent = PPO.load(find_agent_checkpoint(env_cfg["models"]), device=device)

	print(f"Rolling out policy on {env_cfg['env_name']} (seed={SEED})...")
	real_frames, latents, props, actions, success = rollout_real(env_cfg, vq, lstm, agent, SEED, MAX_STEPS)
	print(f"Episode length: {len(actions)} steps, success={success}")

	print("Imagining the same rollout with the world model...")
	imagined_imgs, init_len = imagine_rollout(vq, lstm, latents, props, actions, INIT_LEN, device)

	print("Re-rendering the same episode at high resolution...")
	hq_frames = render_hq_rollout(env_cfg, SEED, actions, HQ_SIZE)

	video1 = [pil_pixelated(f, UPSCALE_SIZE) for f in real_frames]
	save_video(video1, os.path.join(OUT_DIR, f"rollout_real_pixelated_{ENV_NAME}.mp4"))

	context = [pil_pixelated(f, UPSCALE_SIZE) for f in real_frames[:init_len + 1]]
	imagined = [pil_pixelated(tensor_to_pil(imagined_imgs[i]), UPSCALE_SIZE) for i in range(imagined_imgs.shape[0])]
	save_video(context + imagined, os.path.join(OUT_DIR, f"rollout_imagined_{ENV_NAME}.mp4"))

	save_video(hq_frames, os.path.join(OUT_DIR, f"rollout_hq_{ENV_NAME}.mp4"))

	make_paper_figure(real_frames, imagined_imgs, init_len,
					   os.path.join(OUT_DIR, f"rollout_paper_figure_{ENV_NAME}.png"), ENV_NAME,
					   n_columns=N_FIGURE_COLUMNS, step=FIGURE_STEP,
					   hq_frames=hq_frames if FIGURE_INCLUDE_HQ else None)


if __name__ == "__main__":
	main()
