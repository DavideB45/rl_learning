"""
Imagined rollout for the real (soft/pneumatic) robot, purely from the world model: no live
hardware or simulator involved. A previously-recorded episode (images + actions +
proprioception, already on disk under data/real-soft/imgs/) is used only to seed the first
INIT_LEN frames and to supply the action sequence; everything after that is the LSTM world
model predicting forward on its own, decoded back to images through the VQ-VAE.

Produces the same two outputs as rollout_media.py, minus anything that needs a live
env/simulator (no high-quality re-render is possible here - there's no simulator to replay
actions in, and the recorded frames are the only resolution that exists):
  1. a "real" video (the recorded frames, pixelated-upsampled) and an "imagined" video
     (context frames + the world model's own continuation)
  2. a paper-ready figure: sampled recorded frames with the imagined counterpart below them

Kept independent from rollout_media.py / Meta-World on purpose, so it has no dependency on
metaworld/gymnasium/stable-baselines3 and can run wherever just the checkpoints + recorded
data are available.
"""
import os
import sys
import json

import numpy as np
import torch
import torchvision
from PIL import Image
import imageio
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device
from helpers.data import get_data_path
import helpers.model_loader as model_loader
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from global_var import REAL, REAL_DATA_DIR, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, HIDDEN_DIM, SMOOTH, INIT_LEN

# ==========================================
# CONFIGURATION
# ==========================================
EXP_ID = 109   # checkpoint folder under data/real-soft/models/ (also: 2, 105, 109 are available)
ROUND = 109    # recorded round to pull the seed episode from, data/real-soft/imgs/{SPLIT}/round_<ROUND>
SPLIT = "tr"   # "tr" or "vl"
EPISODE = 80    # which episode within that round/split to imagine from

UPSCALE_SIZE = 512  # pixelated upsample target for both videos and the figure
FPS = 10

FIGURE_STEP = None     # steps between consecutive sampled frames; overrides N_FIGURE_COLUMNS when set
N_FIGURE_COLUMNS = 6    # used only when FIGURE_STEP is None: this many evenly-spaced frames

OUT_DIR = "images"

# The real-soft checkpoints were all trained with prop_dim == a_size == 3 (REAL['a_size']),
# not the current global PROP_SIZE (8, bumped for a later data format) - see the
# prop_dim/a_size mismatch worked out for the Meta-World checkpoints in rollout_media.py.
# Unlike those, the real-soft checkpoints already use the current merge_fc (3*hidden_dim),
# so no legacy architecture shim is needed here - just the right prop_dim.
PROP_SIZE = REAL["a_size"]


# ==========================================
# DATA LOADING (no live env - straight from disk)
# ==========================================
@torch.no_grad()
def load_episode_from_disk(img_dir, split, round_id, episode, vq, device):
	"""
	Loads one previously-recorded episode, encoding each frame through the VQ-VAE, in the
	same (frames, latents, props, actions) shapes rollout_media.rollout_real() returns, so
	it can be fed straight into imagine_rollout()/make_paper_figure().
	"""
	base = get_data_path(img_dir, split == "tr", round_id)
	with open(os.path.join(base, "action_reward_data.json")) as f:
		apr = json.load(f)
	actions = apr["actions"][episode]
	props_raw = apr["proprioception"][episode]  # T+1 entries: initial state + one per action
	T = len(actions)

	to_tensor = torchvision.transforms.ToTensor()
	frames, latents, props = [], [], []
	for step in range(T + 1):
		frame = Image.open(os.path.join(base, f"img_{episode}_{step}.png")).convert("RGB")
		frames.append(frame)
		t_img = to_tensor(frame).unsqueeze(0).to(device)
		_, lat, _ = vq.quantize(vq.encode(t_img))
		latents.append(lat.cpu())
		props.append(torch.tensor(props_raw[step], dtype=torch.float32).view(1, 1, -1))
	return frames, latents, props, actions


# ==========================================
# WORLD MODEL ROLLOUT (identical logic to rollout_media.imagine_rollout)
# ==========================================
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


# ==========================================
# MEDIA HELPERS
# ==========================================
def pil_pixelated(frame, size) -> Image.Image:
	if not isinstance(frame, Image.Image):
		frame = Image.fromarray(np.asarray(frame))
	frame = frame.convert("RGB")
	# the real-robot camera pipeline stores frames as BGR without converting to RGB first,
	# and the VQ-VAE was trained on that same (consistently wrong) channel order, so encoding
	# stays untouched - this swap only happens here, at the final step before display, both
	# for recorded frames and for tensor_to_pil's decoded output (which all flows through here).
	arr = np.array(frame)[:, :, ::-1]
	return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
	arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
	return Image.fromarray(arr)


def save_video(frames, path, fps=FPS):
	imageio.mimsave(path, [np.array(f) for f in frames], fps=fps)
	print(f"Saved {path}")


def make_paper_figure(real_frames, imagined_imgs, init_len, out_path, env_label,
					   n_columns=6, step=None, cell_size=512):
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

	row_defs = [("Recorded", "#4e75a4"), ("World model", "#e34948")]
	fig, axes = plt.subplots(2, n_columns, figsize=(2.2 * n_columns, 4.6))
	if n_columns == 1:
		axes = axes[:, None]

	for col, i in enumerate(idx):
		t = init_len + 1 + i  # absolute rollout step this column corresponds to
		images = [pil_pixelated(real_frames[t], cell_size), pil_pixelated(tensor_to_pil(imagined_imgs[i]), cell_size)]
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

	fig.suptitle(f"Recorded vs. imagined rollout — {env_label}", fontsize=14, fontweight="bold")
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

	env_cfg = dict(REAL)
	env_cfg["models"] = f"{REAL_DATA_DIR}models/{EXP_ID}/"
	print(f"Using checkpoint at {env_cfg['models']}")

	vq = load_vq_vae(env_cfg, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, smoothing, device)
	model_loader.PROP_SIZE = PROP_SIZE
	lstm = load_lstm_quantized(env_cfg, vq, device, HIDDEN_DIM, smoothing, False, False)

	print(f"Loading recorded episode {EPISODE} from {SPLIT}/round_{ROUND}...")
	real_frames, latents, props, actions = load_episode_from_disk(
		env_cfg["img_dir"], SPLIT, ROUND, EPISODE, vq, device
	)
	print(f"Episode length: {len(actions)} steps")

	print("Imagining the same rollout with the world model...")
	imagined_imgs, init_len = imagine_rollout(vq, lstm, latents, props, actions, INIT_LEN, device)

	exp_name = {
		101: "Counterclockwise, teleoperated init",
		105: "Clockwise, random init",
		109: "Far gear, teleoperated init",
	}
	tag = f"real-robot ({exp_name.get(EXP_ID, f'run {EXP_ID}')})"

	video1 = [pil_pixelated(f, UPSCALE_SIZE) for f in real_frames]
	save_video(video1, os.path.join(OUT_DIR, f"rollout_real_pixelated_{tag}.mp4"))

	context = [pil_pixelated(f, UPSCALE_SIZE) for f in real_frames[:init_len + 1]]
	imagined = [pil_pixelated(tensor_to_pil(imagined_imgs[i]), UPSCALE_SIZE) for i in range(imagined_imgs.shape[0])]
	save_video(context + imagined, os.path.join(OUT_DIR, f"rollout_imagined_{tag}.mp4"))

	make_paper_figure(real_frames, imagined_imgs, init_len,
					   os.path.join(OUT_DIR, f"rollout_paper_figure_{tag}.png"), tag,
					   n_columns=N_FIGURE_COLUMNS, step=FIGURE_STEP)


if __name__ == "__main__":
	main()
