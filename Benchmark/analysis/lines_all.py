import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==== CONFIG ====
ROOT_FOLDER = "./data/button-press/full_experiments"
EVAL_EPISODES = 10
EVAL_FREQUENCY = 5000

BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_INTERVAL = 95
SMOOTH_WINDOW = 5
# =================


def bootstrap_ci(data, n_bootstrap=5000, ci=95):
	"""
	Compute bootstrap confidence interval.

	Parameters
	----------
	data : np.ndarray
		1D array of values.
	n_bootstrap : int
		Number of bootstrap samples.
	ci : int
		Confidence interval percentage.

	Returns
	-------
	mean, lower, upper
	"""
	# if len(data) > 2:
	# 	data = np.sort(data)
	# 	data = data[1:-1]
	print(data)
	data = np.array(data, dtype=float)
	data = data[~np.isnan(data)]
	bootstrap_means = []
	for _ in range(n_bootstrap):
		sample = np.random.choice(data, size=len(data), replace=True)
		bootstrap_means.append(np.mean(sample))
	lower = np.percentile( bootstrap_means, (100 - ci) / 2 )
	upper = np.percentile( bootstrap_means, 100 - (100 - ci) / 2 )
	return np.mean(data), lower, upper


def process_single_run(file_path):
	df = pd.read_csv(file_path)

	df["success"] = df["success"].astype(int)

	# Group every evaluation block
	df["eval_id"] = np.arange(len(df)) // EVAL_EPISODES
	grouped = df.groupby("eval_id")["success"].mean().reset_index()

	grouped["steps"] = grouped["eval_id"] * EVAL_FREQUENCY

	return grouped[["steps", "success"]]


def process_algorithm(folder_path):
	files = sorted(glob.glob(os.path.join(folder_path, "res_*.csv")))

	runs = []

	for f in files:
		run_df = process_single_run(f)
		run_df = run_df.rename(columns={"success": os.path.basename(f)})
		runs.append(run_df)

	# ==========================================================
	# Remove best and worst runs (optional)
	# ==========================================================
	avgs = [run_df.iloc[:, 1].mean() for run_df in runs]
	min_idx = avgs.index(min(avgs))
	max_idx = avgs.index(max(avgs))
	if len(runs) > 2:
		if min_idx != max_idx:
			runs.pop(max(min_idx, max_idx))
			runs.pop(min(min_idx, max_idx))
			#remove best and worst but per step ( so when sampling)
		else:
			runs.pop(min_idx)

	# ==========================================================
	# Merge all runs
	# ==========================================================
	merged = runs[0]
	for df in runs[1:]:
		merged = pd.merge(merged, df, on="steps", how="inner")
	success_cols = merged.columns.drop("steps")

	# ==========================================================
	# Bootstrap confidence intervals
	# ==========================================================
	means = []
	lowers = []
	uppers = []

	for _, row in merged.iterrows():
		values = row[success_cols].values.astype(float)

		mean, lower, upper = bootstrap_ci(
			values,
			n_bootstrap=BOOTSTRAP_SAMPLES,
			ci=CONFIDENCE_INTERVAL
		)

		means.append(mean)
		lowers.append(lower)
		uppers.append(upper)

	merged["mean"] = means
	merged["lower"] = lowers
	merged["upper"] = uppers

	# ==========================================================
	# Optional smoothing
	# ==========================================================
	merged["mean"] = merged["mean"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
	merged["lower"] = merged["lower"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
	merged["upper"] = merged["upper"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
	return merged[["steps", "mean", "lower", "upper"]]


def main():
	plt.figure(figsize=(10, 10))

	algo_folders = sorted([
		f for f in os.listdir(ROOT_FOLDER)
		if os.path.isdir(os.path.join(ROOT_FOLDER, f))
	])

	# ==========================================================
	# Remove folders you don't want
	# ==========================================================
	for folder in [
		#"default",
		"no_mask",
		"models",
		#"contraction_loss",
		"no_kl",
		"no_kl_no_rew",
		"no_reward",
		"no_ds",
		"no_mask_no_double_step",
	]:
		if folder in algo_folders:
			algo_folders.remove(folder)

	# ==========================================================
	# Plot each algorithm
	# ==========================================================
	for algo in algo_folders:

		n_runs = len(
			glob.glob(
				os.path.join(ROOT_FOLDER, algo, "res_*.csv")
			)
		)

		print(algo, n_runs)

		folder_path = os.path.join(ROOT_FOLDER, algo)

		df = process_algorithm(folder_path)


		plt.plot(df["steps"], df["mean"], label=algo, linewidth=4)

		# Bootstrap confidence interval
		plt.fill_between(
			df["steps"],
			df["lower"],
			df["upper"],
			alpha=0.15
		)

	# ==========================================================
	# Styling
	# ==========================================================
	plt.xlabel("Environment Steps", fontsize=14)
	plt.ylabel("Success Rate", fontsize=14)
	plt.title("Drawer Open Success Rate\n(Mean with Bootstrap 95% CI)",fontsize=16)
	plt.legend(fontsize=11)
	plt.grid(True, alpha=0.3)
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig('fig_all.png', dpi=400)


if __name__ == "__main__":
	main()