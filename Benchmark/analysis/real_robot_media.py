"""
Thesis figures for real-robot episodes, the real-world counterpart of rollout_media.py's
paper figure. Each episode in VIDEO_DIR was recorded from three frame-aligned views:
  - episode_<N>.mp4          the original HD camera frame
  - episode_<N>_cropped.mp4  the same view at the 64x64 resolution the models actually see
                             (stored upscaled to 512x512)
  - episode_<N>_aruco.mp4    the top-down camera tracking the ArUco marker on the gear
For every episode, frames are sampled at a regular interval and the three views are stacked
per sampled step. The tentacle oscillates, so many more frames are shown than in the
simulated figures; to keep the figure page-sized, the sequence wraps into several strips.
"""
import os

import numpy as np
import imageio.v3 as iio
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_DIR = "to_show"
EPISODES = [1, 31, 61]
OUT_DIR = "images"

FIGURE_STEP = 10         # steps between consecutive sampled frames
COLUMNS_PER_STRIP = None # sampled frames per strip before wrapping; None = everything on one line
CELL_SIZE = 512
NATIVE_SIZE = 64         # resolution the cropped view was captured at, before upscaling

ROW_DEFS = [
	("", "HD camera", "#4e9a51"),
	# ("_cropped", "Model input\n(64×64)", "#4e75a4"),  # dropped: too busy next to the HD view
	("_aruco", "Top view\n(ArUco)", "#4a3aa7"),
]


# ==========================================
# FRAME HELPERS
# ==========================================
ANGLE_OVERLAY_BOX = (0, 0, 110, 40)  # where the tracker drew the "<angle> deg" text in the top view
ANGLE_OVERLAY_SCALE = 2


def center_square(frame: Image.Image) -> Image.Image:
	"""The top-view camera is 4:3; crop the middle square (the gear sits roughly centered)
	so every cell in the grid has the same aspect ratio."""
	w, h = frame.size
	s = min(w, h)
	left, top = (w - s) // 2, (h - s) // 2
	return frame.crop((left, top, left + s, top + s))


def aruco_square(frame: Image.Image) -> Image.Image:
	"""center_square would cut off the tracker's angle readout in the top-left corner, so
	paste that corner back onto the crop (it lands on plain background either way),
	enlarged by ANGLE_OVERLAY_SCALE so it stays legible at figure cell size."""
	square = center_square(frame)
	overlay = frame.crop(ANGLE_OVERLAY_BOX)
	overlay = overlay.resize((overlay.width * ANGLE_OVERLAY_SCALE, overlay.height * ANGLE_OVERLAY_SCALE),
							 Image.LANCZOS)
	square.paste(overlay, (0, 0))
	return square


def clean_pixelated(frame: Image.Image, size) -> Image.Image:
	"""The 64x64 view was saved nearest-upscaled, then video compression blurred the block
	edges; averaging back down to the native grid and re-upscaling with nearest restores
	crisp pixels."""
	return frame.resize((NATIVE_SIZE, NATIVE_SIZE), Image.BOX).resize((size, size), Image.NEAREST)


def load_view(episode, suffix):
	frames = iio.imread(os.path.join(VIDEO_DIR, f"episode_{episode}{suffix}.mp4"))
	return [Image.fromarray(f).convert("RGB") for f in frames]


def prepare(frame, suffix, size):
	if suffix == "_cropped":
		return clean_pixelated(frame, size)
	square = aruco_square(frame) if suffix == "_aruco" else center_square(frame)
	return square.resize((size, size), Image.LANCZOS)


# ==========================================
# FIGURE
# ==========================================
def make_episode_figure(episode, out_path, step=FIGURE_STEP, columns_per_strip=COLUMNS_PER_STRIP,
						cell_size=CELL_SIZE):
	views = {suffix: load_view(episode, suffix) for suffix, _, _ in ROW_DEFS}
	n_frames = min(len(v) for v in views.values())
	idx = list(range(0, n_frames, step))
	columns_per_strip = columns_per_strip or len(idx)

	n_strips = int(np.ceil(len(idx) / columns_per_strip))
	n_rows = len(ROW_DEFS)
	fig = plt.figure(figsize=(2.2 * columns_per_strip, (2.2 * n_rows + 0.3) * n_strips + 0.3),
					 layout="constrained")
	# one subfigure per strip, so consecutive strips are visibly separated
	strip_figs = fig.subfigures(n_strips, 1, squeeze=False)[:, 0]
	axes = [sf.subplots(n_rows, columns_per_strip, squeeze=False) for sf in strip_figs]

	for k in range(n_strips * columns_per_strip):
		strip, col = divmod(k, columns_per_strip)
		for r, (suffix, label, color) in enumerate(ROW_DEFS):
			ax = axes[strip][r, col]
			ax.set_xticks([])
			ax.set_yticks([])
			if k >= len(idx):
				ax.axis("off")  # unused cells in a partially filled last strip
				continue
			t = idx[k]
			ax.imshow(prepare(views[suffix][t], suffix, cell_size), interpolation="nearest")
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_color(color)
				spine.set_linewidth(2)
			if r == 0:
				ax.set_title(f"t = {t}", fontsize=11)
			if col == 0:
				ax.set_ylabel(label, fontsize=12, fontweight="bold", color=color)

	fig.suptitle(f"Real robot rollout — episode {episode}", fontsize=14, fontweight="bold")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {out_path}")


def main():
	os.makedirs(OUT_DIR, exist_ok=True)
	for episode in EPISODES:
		make_episode_figure(episode, os.path.join(OUT_DIR, f"real_robot_episode_{episode}.png"))


if __name__ == "__main__":
	main()
