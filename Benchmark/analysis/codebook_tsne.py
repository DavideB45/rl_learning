"""
Same single-run trace as analysis/codebook_dynamics.py, but with t-SNE instead of
ISOMAP for the 2D projection - kept as a separate file so the two projections can
be generated and compared side by side on the exact same episode.

t-SNE only tries to preserve *local* neighborhoods, not global/geodesic distances,
which is precisely why the tutor suggested ISOMAP instead for justifying an MSE
loss: on a t-SNE plot, two clusters being far apart doesn't reliably mean their
codes are far apart in the original embedding space. This file exists to make
that difference visible, not to replace codebook_dynamics.py's ISOMAP version.

For the two "just numbers" measurements (fraction of unchanged code, consecutive-
vs-random-pair distance), see codebook_dynamics.py - those don't depend on the
2D projection at all, so they aren't duplicated here.

Like codebook_dynamics.py, data comes from a LIVE rollout of the actual trained
checkpoint (VQ-VAE + LSTM + PPO agent), not from stored images on disk - a
data-collection round can predate the checkpoint being analyzed, so reading
stored PNGs risks silently encoding images with the wrong VQ-VAE.

Pick the task with --task (window-open / drawer-open / button-press-td) - see
codebook_dynamics.py's module docstring for why only these three are supported.

Run from the Benchmark/ directory: python analysis/codebook_tsne.py
"""
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from stable_baselines3.ppo import PPO

from helpers.general import best_device
import envs.wrapper as wrapper_module
from helpers.model_loader import load_vq_vae
from global_var import CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, HIDDEN_DIM, SMOOTH
from analysis.rollout_media import LegacyMetaWrapEnv, load_legacy_lstm, resolve_models_dir, find_agent_checkpoint
from analysis.codebook_dynamics import (
	live_rollout_episode, collapse_consecutive_duplicates,
	_ENV_CONFIGS, _ENV_ROOTS, _ENV_PROP_SIZE,
)

TIME_CMAP = "viridis"  # perceptually-uniform sequential, already used elsewhere in this repo


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


def plot_run_trace(latents: np.ndarray, perplexity: float, seed: int, env_label: str, out_dir: str):
	"""
	latents: (T+1, D, W, W) quantized embeddings for one live-rolled-out episode, in order.
	Mirrors ../WorldModelExp/analysis/vae_tsne.py's two-panel layout (embedding
	trace | distance over time), and codebook_dynamics.py's ISOMAP version -
	only the projection method differs.
	"""
	setup_style()
	T = latents.shape[0]
	flat = latents.reshape(T, -1)  # (T+1, D*W*W) - one point per frame, like vae_tsne.py

	# Collapsed the same way as the ISOMAP version: ~2 out of every 3 frames here
	# are an exact repeat of the previous one (the whole-frame code didn't change),
	# so without this a held state is drawn as many overlapping points with many
	# overlapping zero-length segments converging on one dot.
	reduced, first_t, dwell = collapse_consecutive_duplicates(flat)
	perplexity = min(perplexity, reduced.shape[0] - 1)
	# init="random" instead of the default "pca": with fewer samples than
	# dimensions here (a few dozen retained states, D*W*W-dim vectors),
	# sklearn's randomized-SVD PCA init hits a numerical instability (it fails
	# the same way even on plain random data of this shape) - random init sidesteps it.
	coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="random").fit_transform(reduced)

	dist_mean = np.abs(np.diff(flat, axis=0)).mean(axis=1)  # (T,) mean |delta| per dim
	dist_max = np.abs(np.diff(flat, axis=0)).max(axis=1)    # (T,) largest single-dim jump

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

	# Segments colored by time, same as the ISOMAP version: a later revisit of an
	# already-visited state still lands on/near the same spot, but its incoming and
	# outgoing segments are visibly a different color instead of looking tangled.
	norm = plt.Normalize(0, T - 1)
	segments = np.stack([coords[:-1], coords[1:]], axis=1)
	seg_times = (first_t[:-1] + first_t[1:]) / 2
	lc = LineCollection(segments, cmap=TIME_CMAP, norm=norm, linewidth=1.2, alpha=0.8, zorder=1)
	lc.set_array(seg_times)
	ax1.add_collection(lc)
	sc = ax1.scatter(coords[:, 0], coords[:, 1], c=first_t, cmap=TIME_CMAP, norm=norm,
					  s=30 + 40 * np.sqrt(dwell), edgecolors="white", linewidths=0.5, zorder=2)
	ax1.autoscale_view()
	fig.colorbar(sc, ax=ax1, label="Time step")
	ax1.set_title(f"t-SNE trace of one episode's latent — {env_label}\n(point size = how long that state was held)")
	ax1.set_xlabel("t-SNE dim. 1"); ax1.set_ylabel("t-SNE dim. 2")

	steps = np.arange(1, T)
	ax2.plot(steps, dist_mean, marker="o", label="Mean |Δ| per dim.")
	ax2.plot(steps, dist_max, marker="x", label="Max |Δ| over dims.")
	ax2.set_title("Distance between consecutive latents")
	ax2.set_xlabel("Time step"); ax2.set_ylabel("Absolute distance")
	ax2.legend()

	fig.suptitle(f"Live rollout, seed={seed}", fontsize=13)
	plt.tight_layout()
	path = os.path.join(out_dir, f"codebook_tsne_run_{env_label}_seed{seed}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


def main():
	parser = argparse.ArgumentParser(
		description="t-SNE version of codebook_dynamics.py's single-run trace, for comparison against ISOMAP.")
	parser.add_argument("--task", choices=list(_ENV_CONFIGS), default="window-open", help="Which Meta-World checkpoint to roll out.")
	parser.add_argument("--seed", type=int, default=0, help="Seed for the rolled-out episode (also fixes the object/goal placement).")
	parser.add_argument("--max-steps", type=int, default=50, help="Safety cap on steps per rollout (episode usually ends earlier).")
	parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity.")
	parser.add_argument("--out-dir", type=str, default="images")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	device = best_device()
	env = dict(_ENV_CONFIGS[args.task])
	env["models"] = resolve_models_dir(_ENV_ROOTS[args.task])
	env_label = args.task
	smoothing = True if SMOOTH else False

	print(f"Loading VQ-VAE + LSTM + PPO agent for {args.task} (so the rollout matches the checkpoint being analyzed)...")
	vq = load_vq_vae(env, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, smoothing, device)
	vq.eval()
	wrapper_module.PROP_SIZE = _ENV_PROP_SIZE[args.task]
	lstm = load_legacy_lstm(env, vq, device, HIDDEN_DIM, _ENV_PROP_SIZE[args.task], smoothing)
	lstm.eval()
	agent = PPO.load(find_agent_checkpoint(env["models"]), device=device)

	print(f"Rolling out one live episode (seed={args.seed}) to trace with t-SNE...")
	_, run_latents = live_rollout_episode(vq, lstm, agent, env, args.seed, args.max_steps, env_class=LegacyMetaWrapEnv)
	plot_run_trace(run_latents, args.perplexity, args.seed, env_label, args.out_dir)


if __name__ == "__main__":
	main()
