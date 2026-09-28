"""
Checks whether the VQ-VAE codebook has enough temporal/metric structure to justify
training the LSTM dynamics model with an MSE loss on the continuous embeddings
instead of a cross-entropy loss over the discrete code indices.

Data comes from a LIVE rollout of the actual trained checkpoint (VQ-VAE + LSTM +
PPO agent) in the Meta-World simulator, via envs.wrapper.MetaWrapEnv - not from
whatever PNGs happen to be sitting on disk under an env's img_dir. Reading stored
images is risky: a data-collection round can predate the checkpoint being
analyzed (learning_loop.py collects data iteratively across many rounds under
whatever policy existed at the time), so the on-disk images and the loaded
weights can silently be out of sync and produce misleading numbers. Rolling out
the checkpoint itself guarantees the images match the model.

Pick the task with --task (window-open / drawer-open / button-press-td). These
three all use the same "Legacy" LSTM/env classes from analysis/rollout_media.py
(their checkpoints predate a merge_fc architecture change and a PROP_SIZE/a_size
decoupling) - only the environment name and checkpoint files differ between them.
The real-soft checkpoint is deliberately not supported here: it corresponds to
a real physical robot, not a simulator, so there's nothing to roll out live.

Two things, kept deliberately separate:

  1. A single-run ISOMAP trace (analogous to ../WorldModelExp/analysis/vae_tsne.py,
     but with ISOMAP instead of t-SNE, since t-SNE only preserves local
     neighborhoods and can't be trusted to say anything about how large a jump
     between two points actually is - ISOMAP preserves geodesic distances
     globally, which is what "how much does the code move frame to frame"
     needs). One live episode is rolled out, its per-frame latent is flattened
     to a single vector, and the resulting trajectory is projected to 2D and
     plotted next to the raw distance between consecutive frames over time.
  2. Plain numbers (no plots) for the two measurements requested directly:
     the fraction of unchanged code per 4x4 grid position between consecutive
     frames, and the average embedding-space distance between consecutive
     codes vs. random pairs of codes, per grid position.

Run from the Benchmark/ directory: python analysis/codebook_dynamics.py
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.manifold import Isomap

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from stable_baselines3.ppo import PPO

from helpers.general import best_device
import envs.wrapper as wrapper_module
from helpers.model_loader import load_vq_vae
from envs.wrapper import MetaWrapEnv
from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized
from global_var import (
	WINDOW_OPEN, DRAWER_OPEN, BUTTON_TD, PEG_INSERT,
	WINDOWO_DATA_DIR, DRAWERO_DATA_DIR, BUTTON_TD_DATA_DIR, PEG_DATA_DIT,
	CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, HIDDEN_DIM, SMOOTH,
)
from analysis.rollout_media import (
	LegacyMetaWrapEnv, load_legacy_lstm, resolve_models_dir, find_agent_checkpoint,
)

TIME_CMAP = "viridis"  # perceptually-uniform sequential, already used elsewhere in this repo

# These three tasks are the only local checkpoints with a working simulator (the
# fourth local checkpoint, real-soft, is a real physical robot - nothing to roll
# out live there). All three need the "Legacy" LSTM/env classes from rollout_media.py;
# only the env config/root/prop_dim actually differ between them.
_ENV_CONFIGS = {"window-open": WINDOW_OPEN, "drawer-open": DRAWER_OPEN, "button-press-td": BUTTON_TD, "peg-insert": PEG_INSERT}
_ENV_ROOTS = {"window-open": WINDOWO_DATA_DIR, "drawer-open": DRAWERO_DATA_DIR, "button-press-td": BUTTON_TD_DATA_DIR, "peg-insert": PEG_DATA_DIT}
_ENV_PROP_SIZE = {"window-open": 4, "drawer-open": 4, "button-press-td": 4, "peg-insert": 4}


def setup_style():
	sns.set_theme(style="darkgrid")
	plt.rcParams.update({
		"savefig.dpi": 300,
		"font.size": 12,
		"axes.titlesize": 13,
		"axes.titleweight": "bold",
		"axes.labelsize": 12,
		"legend.fontsize": 11,
	})


# ==========================================
# DATA COLLECTION - live rollouts, not stored images
# ==========================================
@torch.no_grad()
def live_rollout_episode(vq: VQVAE, lstm: LSTMQuantized, agent: PPO, env_cfg: dict, seed: int, max_steps: int,
						  env_class=MetaWrapEnv):
	"""
	Rolls out one live episode in the actual Meta-World simulator, driven by the
	trained PPO agent, and records the VQ-VAE's codebook index/quantized latent
	at every step (both already computed inside MetaWrapEnv on every reset/step).

	Returns:
		indices: (T+1, W, W) int64 - the codebook index chosen at each grid position.
		latents: (T+1, D, W, W) float32 - the corresponding quantized embedding vectors.
	"""
	env = env_class(vq, lstm, env_cfg=env_cfg, seed=seed)
	obs, _ = env.reset()

	def current_idx():
		return vq.quantizer.onehot_from_vec(env.current_latent).argmax(dim=1).cpu().numpy()

	idx_steps = [current_idx()]
	lat_steps = [env.current_latent.cpu().numpy()]
	for _ in range(max_steps):
		action, _ = agent.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		idx_steps.append(current_idx())
		lat_steps.append(env.current_latent.cpu().numpy())
		if terminated or truncated:
			break
	env.close()
	return np.concatenate(idx_steps, axis=0), np.concatenate(lat_steps, axis=0)


def live_rollout_episodes(vq: VQVAE, lstm: LSTMQuantized, agent: PPO, env_cfg: dict, seeds, max_steps: int,
						   env_class=MetaWrapEnv):
	"""Rolls out several live episodes (each independent, see live_rollout_episode)."""
	all_indices, all_latents = [], []
	for seed in tqdm(seeds, desc="Rolling out live episodes"):
		idx, lat = live_rollout_episode(vq, lstm, agent, env_cfg, seed, max_steps, env_class=env_class)
		all_indices.append(idx)
		all_latents.append(lat)
	return all_indices, all_latents


# ==========================================
# PART 1: single-run ISOMAP trace
# ==========================================
def collapse_consecutive_duplicates(flat: np.ndarray):
	"""
	Collapses runs of exactly-repeated consecutive rows (the whole-frame code
	just didn't change) into a single node each. Real data is dominated by such
	runs (roughly 2 out of every 3 frames in a typical episode here), so a plot
	that connects every raw frame ends up drawing dozens of exact-zero-length
	segments on top of each other - visually, a single held state looks like a
	hub with many lines converging on it, even though the path itself is a
	perfectly ordinary sequence. Collapsing first fixes that at the source
	instead of trying to detangle it visually afterwards.

	Returns:
		reduced (N, F): one row per distinct held state, in temporal order.
		first_t (N,): the original timestep each retained row first appeared at.
		dwell (N,): how many consecutive frames that state was held for.
	"""
	T = flat.shape[0]
	first_t, dwell = [0], [1]
	for t in range(1, T):
		if np.array_equal(flat[t], flat[first_t[-1]]):
			dwell[-1] += 1
		else:
			first_t.append(t)
			dwell.append(1)
	first_t = np.array(first_t)
	return flat[first_t], first_t, np.array(dwell)


def plot_run_trace(latents: np.ndarray, n_neighbors: int, env_label: str, seed: int, out_dir: str):
	"""
	latents: (T+1, D, W, W) quantized embeddings for one live-rolled-out episode, in order.
	Mirrors vae_tsne.py's two-panel layout (embedding trace | distance over
	time), swapping t-SNE for ISOMAP.
	"""
	setup_style()
	T = latents.shape[0]
	flat = latents.reshape(T, -1)  # (T+1, D*W*W) - one point per frame, like vae_tsne.py

	reduced, first_t, dwell = collapse_consecutive_duplicates(flat)
	n_neighbors = min(n_neighbors, reduced.shape[0] - 1)
	coords = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(reduced)

	dist_mean = np.abs(np.diff(flat, axis=0)).mean(axis=1)  # (T,) mean |delta| per dim
	dist_max = np.abs(np.diff(flat, axis=0)).max(axis=1)    # (T,) largest single-dim jump

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

	# Segments are colored by time (same colormap/scale as the points), instead of a
	# flat gray line: a later revisit of an already-visited state still lands on/near
	# the same spot, but its incoming and outgoing segments are visibly a different
	# color, making that a legible "the state repeated" instead of "a tangled point".
	norm = plt.Normalize(0, T - 1)
	segments = np.stack([coords[:-1], coords[1:]], axis=1)
	seg_times = (first_t[:-1] + first_t[1:]) / 2
	lc = LineCollection(segments, cmap=TIME_CMAP, norm=norm, linewidth=1.2, alpha=0.8, zorder=1)
	lc.set_array(seg_times)
	ax1.add_collection(lc)
	# marker area scales with how long that state was held (sqrt so area, not radius, is linear in dwell)
	sc = ax1.scatter(coords[:, 0], coords[:, 1], c=first_t, cmap=TIME_CMAP, norm=norm,
					  s=30 + 40 * np.sqrt(dwell), edgecolors="white", linewidths=0.5, zorder=2)
	ax1.autoscale_view()
	fig.colorbar(sc, ax=ax1, label="Time step")
	ax1.set_title(f"ISOMAP trace of one episode's latent — {env_label}\n(point size = how long that state was held)")
	ax1.set_xlabel("ISOMAP dim. 1"); ax1.set_ylabel("ISOMAP dim. 2")

	steps = np.arange(1, T)
	ax2.plot(steps, dist_mean, marker="o", label="Mean |Δ| per dim.")
	ax2.plot(steps, dist_max, marker="x", label="Max |Δ| over dims.")
	ax2.set_title("Distance between consecutive latents")
	ax2.set_xlabel("Time step"); ax2.set_ylabel("Absolute distance")
	ax2.legend()

	fig.suptitle(f"Live rollout, seed={seed}", fontsize=13)
	plt.tight_layout()
	path = os.path.join(out_dir, f"codebook_isomap_run_{env_label}_seed{seed}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


# ==========================================
# PART 2: plain numbers for the tutor's two measurements
# ==========================================
def compute_unchanged_fraction(indices):
	"""
	Args:
		indices: list of (T_ep+1, W, W) int arrays, one per episode.
	Returns:
		frac (W, W): fraction of consecutive-frame pairs where the code index at
			that position didn't change.
		overall (float): the same fraction pooled over every position.
	"""
	unchanged, total = None, 0
	for ep_idx in indices:
		same = (ep_idx[:-1] == ep_idx[1:])  # (T_ep, W, W) bool
		unchanged = same.sum(axis=0) if unchanged is None else unchanged + same.sum(axis=0)
		total += same.shape[0]
	frac = unchanged / total
	overall = unchanged.sum() / (total * frac.size)
	return frac, overall


def compute_distance_stats(latents, rng, n_random_pairs: int = 20000):
	"""
	For each grid position, compares the Euclidean distance (in the D-dim
	embedding space) between consecutive-frame codes against the distance
	between random pairs drawn from every code that position was ever assigned,
	across the whole dataset.

	Returns a dict of (W, W) arrays: consec_mean, consec_std, random_mean, random_std.
	"""
	D, W = latents[0].shape[1], latents[0].shape[-1]
	n_pos = W * W
	pooled_pos = defaultdict(list)
	consec_pos = defaultdict(list)

	for ep in latents:
		flat = ep.reshape(ep.shape[0], D, n_pos)      # (T_ep+1, D, W*W)
		dist = np.linalg.norm(flat[1:] - flat[:-1], axis=1)  # (T_ep, W*W)
		for p in range(n_pos):
			consec_pos[p].append(dist[:, p])
			pooled_pos[p].append(flat[:, :, p])       # (T_ep+1, D)

	consec_mean = np.zeros((W, W)); consec_std = np.zeros((W, W))
	random_mean = np.zeros((W, W)); random_std = np.zeros((W, W))
	for p in range(n_pos):
		r, c = divmod(p, W)
		d = np.concatenate(consec_pos[p])
		consec_mean[r, c], consec_std[r, c] = d.mean(), d.std()

		pool = np.concatenate(pooled_pos[p], axis=0)  # (N, D)
		n = pool.shape[0]
		i = rng.integers(0, n, size=n_random_pairs)
		j = rng.integers(0, n, size=n_random_pairs)
		keep = i != j
		rd = np.linalg.norm(pool[i[keep]] - pool[j[keep]], axis=1)
		random_mean[r, c], random_std[r, c] = rd.mean(), rd.std()

	return dict(consec_mean=consec_mean, consec_std=consec_std,
				random_mean=random_mean, random_std=random_std)


def print_numeric_summary(frac_unchanged, overall_unchanged, dist_stats):
	W = frac_unchanged.shape[0]
	print("\n=== Fraction of unchanged code between consecutive frames (per 4x4 position) ===")
	for r in range(W):
		row = "  ".join(f"{frac_unchanged[r, c]*100:5.1f}%" for c in range(W))
		print(f"row {r}: {row}")
	print(f"Overall (pooled over all {W*W} positions): {overall_unchanged*100:.2f}%")

	print("\n=== Consecutive vs. random-pair embedding distance (per 4x4 position) ===")
	print(f"{'pos':>6} | {'consecutive':>11} | {'random':>11} | {'ratio':>6}")
	for r in range(W):
		for c in range(W):
			cm, rm = dist_stats["consec_mean"][r, c], dist_stats["random_mean"][r, c]
			print(f"({r},{c}) | {cm:11.4f} | {rm:11.4f} | {cm/rm:6.3f}")
	all_cm, all_rm = dist_stats["consec_mean"].mean(), dist_stats["random_mean"].mean()
	print(f"{'all':>6} | {all_cm:11.4f} | {all_rm:11.4f} | {all_cm/all_rm:6.3f}")


# ==========================================
# MAIN
# ==========================================
def main():
	parser = argparse.ArgumentParser(
		description="Analyzes temporal/metric structure of the VQ-VAE codebook to justify an MSE (vs. CE) dynamics loss. "
					"Data comes from a live rollout of the trained checkpoint, not stored images.")
	parser.add_argument("--task", choices=list(_ENV_CONFIGS), default="window-open", help="Which Meta-World checkpoint to roll out.")
	parser.add_argument("--seed", type=int, default=0, help="Seed for the single episode traced with ISOMAP (also fixes the object/goal placement).")
	parser.add_argument("--n-episodes", type=int, default=20, help="How many freshly-rolled-out episodes to pool for the numeric summary.")
	parser.add_argument("--max-steps", type=int, default=50, help="Safety cap on steps per rollout (episode usually ends earlier).")
	parser.add_argument("--n-neighbors", type=int, default=15, help="ISOMAP neighborhood size.")
	parser.add_argument("--n-random-pairs", type=int, default=20000)
	parser.add_argument("--out-dir", type=str, default="images")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	device = best_device()
	env = dict(_ENV_CONFIGS[args.task])
	env["models"] = resolve_models_dir(_ENV_ROOTS[args.task])
	env_label = args.task
	smoothing = True if SMOOTH else False
	rng = np.random.default_rng(args.seed)

	print(f"Loading VQ-VAE + LSTM + PPO agent for {args.task} (so the rollout matches the checkpoint being analyzed)...")
	vq = load_vq_vae(env, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, smoothing, device)
	vq.eval()
	wrapper_module.PROP_SIZE = _ENV_PROP_SIZE[args.task]
	lstm = load_legacy_lstm(env, vq, device, HIDDEN_DIM, _ENV_PROP_SIZE[args.task], smoothing)
	lstm.eval()
	agent = PPO.load(find_agent_checkpoint(env["models"]), device=device)

	print(f"Rolling out one live episode (seed={args.seed}) to trace with ISOMAP...")
	_, run_latents = live_rollout_episode(vq, lstm, agent, env, args.seed, args.max_steps, env_class=LegacyMetaWrapEnv)
	plot_run_trace(run_latents, args.n_neighbors, env_label, args.seed, args.out_dir)

	print(f"Rolling out {args.n_episodes} live episodes for the numeric summary...")
	seeds = list(range(args.n_episodes))
	indices, latents = live_rollout_episodes(vq, lstm, agent, env, seeds, args.max_steps, env_class=LegacyMetaWrapEnv)
	frac_unchanged, overall_unchanged = compute_unchanged_fraction(indices)
	dist_stats = compute_distance_stats(latents, rng, n_random_pairs=args.n_random_pairs)
	print_numeric_summary(frac_unchanged, overall_unchanged, dist_stats)


if __name__ == "__main__":
	main()
