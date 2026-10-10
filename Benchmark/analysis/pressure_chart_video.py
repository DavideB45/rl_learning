"""
Pressure-vs-time chart video for a single real-robot episode, synchronized frame-for-frame
with episode_<N>.mp4 / _cropped.mp4 / _aruco.mp4 so it can be played side by side with them.

The episode this was built for (episode_81_pressure.mp4, a fresh live recording of the round
105 policy, labeled "81" for where it sits on the CSV training curve - see
plot_real_robot_angle.py - not an index into the historical JSON dataset) was never logged to
action_reward_data.json, so there is no numeric pressure log for it. The only record of the
per-chamber pressure is the live debug overlay RealWorld.render_pressure_window() draws every
frame (3 solid-colored vertical bars, one per chamber, 0 to max_pressure), which got saved
alongside the other views as episode_81_pressure.mp4. This script inverts that drawing: for
each frame, it scans down the known x-column of each bar for the first pixel matching that
chamber's known fill color, which is exactly y_fill from render_pressure_window(), and turns
that back into a 0-100% reading - recovered from the video's pixels since the exact values
were not separately logged, so treat it as a faithful but imperfect (compression-limited)
reconstruction, not a ground-truth log.
"""
import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_VIDEO = "episode_81_pressure.mp4"
OUTPUT_VIDEO = "episode_81_pressure_chart.mp4"
EPISODE_LABEL = "Episode 81"

# Geometry/colors from envs/physical/realWorld.py's RealWorld.render_pressure_window() -
# must match exactly for the pixel readback to be correct.
GAUGE_WIDTH, GAUGE_HEIGHT = 220, 320
MARGIN = 30
BAR_AREA_H = GAUGE_HEIGHT - 2 * MARGIN
BAR_W = (GAUGE_WIDTH - 2 * MARGIN) // 3 - 10
BAR_COLORS_BGR = [(60, 180, 255), (80, 220, 100), (60, 120, 255)]  # one per chamber
COLOR_MATCH_TOL = 60  # max color distance accepted as "this is the bar", compression-tolerant

CHAMBER_LABELS = ["Chamber 1", "Chamber 2", "Chamber 3"]
# Derived from BAR_COLORS_BGR (not a separate hardcoded list) so the lines can never drift
# out of sync with the bar-gauge colors they're meant to match.
LINE_COLORS = ["#{:02x}{:02x}{:02x}".format(r, g, b) for b, g, r in BAR_COLORS_BGR]
LINE_WIDTH = 2
DPI = 200

OUT_SIZE = (2*640, 1.5*480)  # matches episode_81_aruco.mp4's resolution, for easy side-by-side viewing


# ==========================================
# PRESSURE EXTRACTION (invert render_pressure_window's drawing)
# ==========================================
def bar_x_centers():
	return [MARGIN + i * (BAR_W + 10) + BAR_W // 2 for i in range(3)]


def frac_from_column(frame, x, color_bgr):
	column = frame[:, x, :].astype(int)
	dist = np.linalg.norm(column - np.array(color_bgr), axis=1)
	matches = np.where(dist < COLOR_MATCH_TOL)[0]
	y_bottom = GAUGE_HEIGHT - MARGIN
	if len(matches) == 0:
		return 0.0  # no matching pixel: the bar is empty (pressure ~0)
	y_fill = matches.min()
	return float(np.clip((y_bottom - y_fill) / BAR_AREA_H, 0.0, 1.0))


def extract_pressure_series(video_path):
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	xs = bar_x_centers()
	series = []
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		series.append([frac_from_column(frame, x, c) * 100.0 for x, c in zip(xs, BAR_COLORS_BGR)])
	cap.release()
	return np.array(series), fps  # (T, 3), fps


# ==========================================
# CHART VIDEO (growing line, revealed in lockstep with the source video)
# ==========================================
def render_chart_video(pressure_pct, fps, out_path, episode_label):
	import seaborn as sns
	sns.set_theme(style="darkgrid")

	T = pressure_pct.shape[0]
	t = np.arange(T) / fps

	fig, ax = plt.subplots(figsize=(OUT_SIZE[0] / DPI, OUT_SIZE[1] / DPI), dpi=DPI)
	lines = [ax.plot([], [], color=c, linewidth=LINE_WIDTH, label=l)[0] for c, l in zip(LINE_COLORS, CHAMBER_LABELS)]
	dots = [ax.plot([], [], "o", color=c, markersize=6)[0] for c in LINE_COLORS]

	ax.set_xlim(0, t[-1] if t[-1] > 0 else 1)
	ax.set_ylim(-5, 105)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Chamber pressure (%)")
	ax.set_title(f"Chamber Pressure — {episode_label}", fontweight="bold")
	ax.legend(loc="upper left")
	fig.tight_layout()

	frames = []
	for i in range(T):
		for line, dot, series in zip(lines, dots, pressure_pct.T):
			line.set_data(t[:i + 1], series[:i + 1])
			dot.set_data([t[i]], [series[i]])
		fig.canvas.draw()
		buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
		frames.append(buf.copy())
	plt.close(fig)

	imageio.mimsave(out_path, frames, fps=fps)
	print(f"Saved {out_path}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
	print(f"Reading pressure bars from {SOURCE_VIDEO}...")
	pressure_pct, fps = extract_pressure_series(SOURCE_VIDEO)
	print(f"Extracted {pressure_pct.shape[0]} frames at {fps:.1f} fps")
	render_chart_video(pressure_pct, fps, OUTPUT_VIDEO, EPISODE_LABEL)
