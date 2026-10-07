"""
Single-run plot for the real-robot (soft/pneumatic tentacle) training logs res_101.csv,
res_105.csv, res_109.csv.

These are plotted differently from every other single run (see plot_single_run.py): their
`success` column is not trustworthy (it only checks the final rotation at episode end
against a fixed threshold, not the best angle reached - see RealWorld.step()'s comment on
why that's not the same thing as thresholding the reward), so it isn't plotted at all. The
`reward` column IS trustworthy, and for this task it's a direct linear rescaling of the
angle (in radians) the gear was rotated by, so it's converted to degrees and plotted
directly: a more physically meaningful axis than an arbitrary reward unit, which is also why
this gets horizontal reference lines at every 90 degrees up to whatever the policy reached.

== Reward -> angle conversion ==
envs/wrapper.py's SoftWrapEnv always builds RealWorld with rew_multiplier=30.0, and
RealWorld always builds its ArucoRotationTracker with units='rad' (scale=1.0 in
ArucoRotationTracker._UNITS). So every step: reward = reset_reward() * 30.0, in radians.
reset_reward() returns max(0, current_angle - best_angle_so_far) and raises the best angle
to it, so summed over an episode (as the logged `mrew` already is, see
envs/wrapper.py's evaluate_gathering: `tot_rewards[-1] += rew`) it telescopes to exactly the
max angle reached during that episode, in radians, times 30. Hence:
    angle_degrees = mrew * (180 / (pi * 30))
This mirrors plot_single_run.py's statistics (Student-t CI on the pooled, rolling-windowed
episodes) and seaborn darkgrid styling, just with reward rescaled to degrees and the success
panel dropped.
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

# ==========================================
# CONFIGURATION
# ==========================================
RUNS = [
	{"exp_id": 101, "csv": "res_101.csv", "label": "Counterclockwise, teleoperated init", "color": "#008300"},
	{"exp_id": 105, "csv": "res_105.csv", "label": "Clockwise, random init", "color": "#4a3aa7"},
	{"exp_id": 109, "csv": "res_109.csv", "label": "Far gear, teleoperated init", "color": "#eb6834"},
]

# envs/wrapper.py's SoftWrapEnv hardcodes rew_multiplier=30.0; RealWorld's
# ArucoRotationTracker always uses units='rad' (scale=1.0) - see module docstring.
REWARD_MULTIPLIER = 30.0
REWARD_TO_DEGREES = 180.0 / (math.pi * REWARD_MULTIPLIER)

# Evaluation parameters - one CSV row is one episode (EPISODES_PER_EVAL=1), and
# RealWorld/SoftWrapEnv caps every episode at max_steps=150.
EPISODES_PER_EVAL = 1
EPISODE_LENGTH = 150
STEPS_PER_EVAL = EPISODE_LENGTH * EPISODES_PER_EVAL

# Number of consecutive evaluation batches pooled together for smoothing (same idea as
# plot_single_run.py / final_plot.py's rolling mean - pooling also pools their episodes,
# giving a tighter, more honest interval as more evaluations accumulate).
ROLLING_WINDOW = 10
CI = 95

# All 3 runs start with this many steps of untrained/undirected data collection before the
# logged episodes begin, same convention as final_plot.py/plot_single_run.py's warmup shift:
# shifts the x-axis right so it reflects true total environment steps, and anchors the curve
# at (0 steps, 0 degrees) so the warmup phase reads as a flat lead-in instead of being hidden.
WARMUP_STEPS = 1500

REFERENCE_LINE_STEP_DEG = 90  # horizontal guides every this many degrees

OUT_DIR = "images"


# ==========================================
# DATA PROCESSING (reward -> angle, then pooled rolling CI - mirrors plot_single_run.py)
# ==========================================
def load_and_process_data(csv_path):
	df = pd.read_csv(csv_path)
	df["angle_deg"] = df["mrew"] * REWARD_TO_DEGREES

	window_episodes = min(EPISODES_PER_EVAL * ROLLING_WINDOW, len(df))

	# A plain trailing rolling window only has 1, 2, ... episodes in it for the first few
	# points, which makes the CI explode there (n=2 -> t~12.7). Instead, every point pools
	# exactly `window_episodes` episodes: a trailing window, clamped to start at row 0 until
	# enough rows exist (so the first few points share the first full window).
	starts = (np.arange(len(df)) - window_episodes + 1).clip(min=0)
	windows = [slice(s, s + window_episodes) for s in starts]

	def windowed(col, fn):
		return pd.Series([fn(df[col].iloc[w]) for w in windows], index=df.index)

	roll_mean = windowed("angle_deg", lambda x: x.mean())
	roll_std = windowed("angle_deg", lambda x: x.std(ddof=1))
	roll_n = windowed("angle_deg", lambda x: x.count())

	chunk_rows = df.index[EPISODES_PER_EVAL - 1::EPISODES_PER_EVAL]
	chunk_rows = chunk_rows[roll_mean.loc[chunk_rows].notna()]

	result = pd.DataFrame({
		"step": (chunk_rows // EPISODES_PER_EVAL) * STEPS_PER_EVAL,
		"angle_smooth": roll_mean.loc[chunk_rows].values,
		"angle_std": roll_std.loc[chunk_rows].values,
		"angle_n": roll_n.loc[chunk_rows].values,
	})

	# Student-t interval from the pooled episodes' std.
	t_crit = stats.t.ppf(1 - (1 - CI / 100) / 2, df=(result["angle_n"] - 1).clip(lower=1))
	halfwidth = t_crit * result["angle_std"] / np.sqrt(result["angle_n"])
	result["angle_lo"] = result["angle_smooth"] - halfwidth
	result["angle_hi"] = result["angle_smooth"] + halfwidth

	result["step"] += WARMUP_STEPS
	anchor = pd.DataFrame([{
		"step": 0, "angle_smooth": 0.0, "angle_std": 0.0, "angle_n": 1,
		"angle_lo": 0.0, "angle_hi": 0.0,
	}])
	result = pd.concat([anchor, result], ignore_index=True)

	return result


# ==========================================
# PLOTTING
# ==========================================
def plot_results(data, label, color, out_path):
	sns.set_theme(style="darkgrid")
	fig, ax = plt.subplots(figsize=(8, 5))

	ax.plot(data["step"], data["angle_smooth"], color=color, label=label, linewidth=2)
	ax.fill_between(data["step"], data["angle_lo"], data["angle_hi"], color=color, alpha=0.2)

	# Horizontal reference lines every REFERENCE_LINE_STEP_DEG degrees, up to whatever the
	# policy actually reached, so the viewer can read off how many quarter/half/full turns
	# were achieved.
	max_angle = data["angle_smooth"].max()
	n_lines = int(max_angle // REFERENCE_LINE_STEP_DEG) + 1
	for k in range(1, n_lines + 1):
		deg = k * REFERENCE_LINE_STEP_DEG
		ax.axhline(deg, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
		ref_label = f"{deg}°" + (f" ({deg // 360} turn{'s' if deg // 360 != 1 else ''})" if deg % 360 == 0 else "")
		ax.text(data["step"].max(), deg, f"  {ref_label}", va="center", ha="left", fontsize=8, color="gray")

	ax.set_title(f"Reached Angle ({CI}% CI) — {label}", fontweight="bold")
	ax.set_xlabel("Environment Steps")
	ax.set_ylabel("Reached angle (degrees)")
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%d°"))
	ax.legend(loc="upper left")

	plt.tight_layout()
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {out_path}")


def print_final_summary(data, label):
	last = data.iloc[-1]
	print(f"{label}: final reached angle {last['angle_smooth']:.1f}° "
		  f"({CI}% CI [{last['angle_lo']:.1f}, {last['angle_hi']:.1f}]°), "
		  f"max {data['angle_smooth'].max():.1f}°")


if __name__ == "__main__":
	import os
	os.makedirs(OUT_DIR, exist_ok=True)
	for run in RUNS:
		print(f"Processing {run['csv']}...")
		data = load_and_process_data(run["csv"])
		out_path = os.path.join(OUT_DIR, f"real_robot_angle_{run['exp_id']}.png")
		plot_results(data, run["label"], run["color"], out_path)
		print_final_summary(data, run["label"])
