import os
import sys
import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.general import best_device
from vae.vqVae import VQVAE
from global_var import CURRENT_ENV, LATENT_DIM, CODE_DEPTH, CODEBOOK_SIZE
from helpers.data import make_image_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae

# ==========================================
# STYLE (kept consistent with analysis/final_plot.py so every thesis figure
# reads as one system). "default" (KL/flatness loss enabled) and "no_kl"
# reuse the exact colors and labels registered for those experiments in
# analysis/final_plot.py's EXPERIMENT_COLORS / EXPERIMENT_LABELS.
# ==========================================
WITH_KL_TAG = "default"
NO_KL_TAG = "no_kl"
VARIANTS = {
	WITH_KL_TAG: dict(smooth=True, label="Ours", color="#008300"),
	NO_KL_TAG: dict(smooth=False, label="Ours (no KL loss)", color="#e34948"),
}
METRIC_COLORS = {
	"Active codes (%)": "#4e75a4",
	"Normalized entropy (%)": "#eda100",
}


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
# MODEL LOADING
# ==========================================
def try_load_vq_vae(env: dict, smooth: bool, device) -> VQVAE:
	try:
		model = load_vq_vae(env, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, smooth, device)
		model.eval()
		return model
	except (FileNotFoundError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
		print(f"  -> couldn't load checkpoint for smooth={smooth} ({type(e).__name__}: {e}), skipping this variant.")
		return None


# ==========================================
# RECONSTRUCTION / MASK GRID
# ==========================================
def _to_img(t: torch.Tensor) -> np.ndarray:
	t = t.detach()
	if t.dim() == 3:
		t = t.permute(1, 2, 0)
	return t.numpy()[::-1, ::-1]


@torch.no_grad()
def compute_reconstructions(model: VQVAE, images: torch.Tensor):
	recon, _, _ = model.forward(images)
	z = model.encode(images)
	decoded = model.decoder(z)
	mask = model.pred_mask(decoded)                     # P(background)
	foreground_mask = (1 - mask).squeeze(1).cpu()        # P(foreground)
	foreground = (model.pred_robot(decoded) * (1 - mask)).cpu()
	background = torch.sigmoid(model.backgorund.data.cpu())
	return recon.cpu(), foreground_mask, foreground, background


def plot_reconstruction_grid(images, recon, fg_mask, foreground, tag, label, color, num_images, out_dir):
	setup_style()
	rows = [
		("Input", images, None),
		("Reconstruction", recon, None),
		("Foreground Mask", fg_mask, "viridis"),
		("Foreground", foreground, None),
	]
	n = min(num_images, images.size(0))
	fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 1.9 * len(rows)))
	if n == 1:
		axes = axes[:, None]
	for r, (row_label, tensor, cmap) in enumerate(rows):
		for c in range(n):
			ax = axes[r, c]
			ax.imshow(_to_img(tensor[c]), cmap=cmap)
			ax.set_xticks([])
			ax.set_yticks([])
			for spine in ax.spines.values():
				spine.set_visible(False)
			if c == 0:
				ax.set_ylabel(row_label, fontsize=12, fontweight="bold")
	fig.suptitle(f"VQ-VAE Reconstructions — {label}", fontsize=14, fontweight="bold", color=color)
	plt.tight_layout()
	path = os.path.join(out_dir, f"vq_reconstruction_{tag}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


def plot_background(background, tag, label, color, out_dir):
	setup_style()
	fig, ax = plt.subplots(figsize=(3, 3))
	ax.imshow(_to_img(background))
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.set_title(f"Learned Background — {label}", fontsize=12, fontweight="bold", color=color)
	plt.tight_layout()
	path = os.path.join(out_dir, f"vq_background_{tag}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


# ==========================================
# CODEBOOK USAGE
# ==========================================
@torch.no_grad()
def compute_codebook_usage(model: VQVAE, loader, device) -> dict:
	counts = np.zeros(model.codebook_size, dtype=np.int64)
	total_mse = 0.0
	for batch, _ in tqdm(loader, desc="Evaluating codebook usage"):
		batch = batch.to(device)
		recon, _, indexes = model.forward(batch)
		total_mse += model.reconstruction_loss(batch, recon).item()
		np.add.at(counts, indexes.cpu().numpy().flatten(), 1)
	active = int((counts > 0).sum())
	probs = counts / counts.sum()
	nonzero = probs[probs > 0]
	norm_entropy = -(nonzero * np.log(nonzero)).sum() / np.log(model.codebook_size)
	return {
		"counts": counts,
		"avg_mse": total_mse / len(loader),
		"active_codes": active,
		"active_pct": active / model.codebook_size * 100,
		"norm_entropy": norm_entropy,
	}


def plot_codebook_single(label, color, stats, tag, out_dir):
	setup_style()
	fig, ax = plt.subplots(figsize=(8, 5))
	sorted_counts = np.sort(stats["counts"])[::-1]
	ranks = np.arange(len(sorted_counts))  # 0-indexed, same convention as the original script
	ax.bar(ranks, sorted_counts, color=color, width=1.0, edgecolor="white", linewidth=0.3)
	ax.set_ylim(bottom=0)
	ax.set_xlim(-0.5, len(sorted_counts) - 0.5)
	active = stats["active_codes"]
	if 0 < active < len(sorted_counts):
		ax.axvline(active - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
		ax.text(active, ax.get_ylim()[1] * 0.97, f" {active} active codes", va="top", ha="left", fontsize=10)
	ax.set_xlabel("Codebook index (rank-sorted, most → least used)")
	ax.set_ylabel("Usage count")
	ax.set_title(
		f"Codebook Usage — {label}\n"
		f"{stats['active_pct']:.1f}% active codes, normalized entropy {stats['norm_entropy']*100:.1f}%",
		color=color,
	)
	plt.tight_layout()
	path = os.path.join(out_dir, f"codebook_usage_{tag}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


def plot_codebook_comparison(stats_by_tag: dict, env_name: str, out_dir: str):
	setup_style()
	fig, axes = plt.subplots(1, 2, figsize=(13, 5))

	# Panel 1: rank-sorted usage frequency, linear scale, pinned at 0 so a dead
	# code (count == 0) is unambiguous instead of just vanishing like on a log axis.
	# Step lines (instead of two overlapping bar series) keep both variants readable.
	max_len = max(len(s["counts"]) for _, _, s in stats_by_tag.values())
	for tag, (label, color, stats) in stats_by_tag.items():
		sorted_counts = np.sort(stats["counts"])[::-1]
		ranks = np.arange(len(sorted_counts))
		axes[0].fill_between(ranks, sorted_counts, step="post", color=color, alpha=0.15)
		axes[0].step(ranks, sorted_counts, where="post", color=color, linewidth=2, label=label)
	axes[0].set_ylim(bottom=0)
	axes[0].set_xlim(0, max_len - 1)
	axes[0].set_xlabel("Codebook index (rank-sorted, most → least used)")
	axes[0].set_ylabel("Usage count")
	axes[0].set_title("Codebook Usage Frequencies")
	axes[0].legend()

	# Panel 2: active-code % and normalized entropy, grouped by variant
	labels = [v[0] for v in stats_by_tag.values()]
	colors = [v[1] for v in stats_by_tag.values()]
	active_pct = [v[2]["active_pct"] for v in stats_by_tag.values()]
	norm_entropy = [v[2]["norm_entropy"] * 100 for v in stats_by_tag.values()]

	x = np.arange(len(labels))
	width = 0.35
	axes[1].bar(x - width / 2, active_pct, width, color=METRIC_COLORS["Active codes (%)"], label="Active codes (%)")
	axes[1].bar(x + width / 2, norm_entropy, width, color=METRIC_COLORS["Normalized entropy (%)"], label="Normalized entropy (%)")
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(labels)
	for tick, c in zip(axes[1].get_xticklabels(), colors):
		tick.set_color(c)
		tick.set_fontweight("bold")
	axes[1].set_ylim(0, 105)
	axes[1].set_ylabel("%")
	axes[1].set_title("Codebook Utilization Summary")
	axes[1].legend()

	plt.tight_layout()
	path = os.path.join(out_dir, f"codebook_usage_comparison_{env_name}.png")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {path}")
	return path


# ==========================================
# MAIN
# ==========================================
def main():
	parser = argparse.ArgumentParser(description="Generate thesis-ready VQ-VAE evaluation plots.")
	parser.add_argument("--num-images", type=int, default=6, help="Number of samples in the reconstruction grid.")
	parser.add_argument("--out-dir", type=str, default="images", help="Directory where plots are saved.")
	parser.add_argument("--images-only", action="store_true",
						 help="Only dump reconstruction/mask/background images, skip the (slow, full-test-set) "
							  "codebook usage/distribution analysis and its plots entirely.")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	device = best_device()
	env = CURRENT_ENV
	print(f"Testing {env['env_name']} VAE model")

	test_loader = make_image_dataloader_safe(get_data_path(env['img_dir'], False, 2))
	sample_images = next(iter(test_loader))[0].to(device)

	stats_by_tag = {}
	for tag, cfg in VARIANTS.items():
		label, color, smooth = cfg["label"], cfg["color"], cfg["smooth"]
		print(f"\n=== {label} (smooth={smooth}) ===")
		model = try_load_vq_vae(env, smooth, device)
		if model is None:
			continue

		# Images are generated and saved first, and independently of the
		# (optional, slower) codebook usage analysis below, so a model that
		# was only ever trained/loaded on its own still gets its pictures.
		recon, fg_mask, foreground, background = compute_reconstructions(model, sample_images)
		plot_reconstruction_grid(sample_images.cpu(), recon, fg_mask, foreground, tag, label, color,
								  num_images=args.num_images, out_dir=args.out_dir)
		plot_background(background, tag, label, color, out_dir=args.out_dir)

		if args.images_only:
			continue

		try:
			stats = compute_codebook_usage(model, test_loader, device)
		except Exception as e:
			print(f"  -> codebook usage analysis failed ({type(e).__name__}: {e}), reconstruction images were still saved.")
			continue
		print(f"Average reconstruction error (MSE): {stats['avg_mse']:.4f}")
		print(f"Active codes: {stats['active_codes']}/{model.codebook_size} ({stats['active_pct']:.2f}%)")
		print(f"Normalized codebook entropy: {stats['norm_entropy']*100:.2f}%")
		plot_codebook_single(label, color, stats, tag, out_dir=args.out_dir)
		stats_by_tag[tag] = (label, color, stats)

	if args.images_only:
		print("\n--images-only set: skipped the codebook usage/distribution analysis and comparison plot.")
	elif len(stats_by_tag) == 2:
		plot_codebook_comparison(stats_by_tag, env['env_name'], out_dir=args.out_dir)
		print("\n-- LaTeX table rows (codebook utilization) --")
		for label, color, stats in stats_by_tag.values():
			print(f"{label} & {stats['active_pct']:.1f} & {stats['norm_entropy']*100:.1f} & {stats['avg_mse']:.3f} \\\\")
	elif not stats_by_tag:
		print("\nNo checkpoints found for either variant (looked for smooth=True/False) - nothing to plot.")
	else:
		print("\nOnly one variant available locally - skipping the KL-smoothing comparison plot.")


if __name__ == "__main__":
	main()
