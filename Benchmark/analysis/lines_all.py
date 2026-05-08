import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==== CONFIG ====
ROOT_FOLDER = "./data/drawer-open/full_experiments"   # folder containing all algorithm folders
EVAL_EPISODES = 5
EVAL_FREQUENCY = 5000
# =================


def process_single_run(file_path):
	df = pd.read_csv(file_path)

	df["success"] = df["success"].astype(int)

	# Group every 5 rows (one evaluation)
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

	# Compute average success for each run
	avgs = [run_df.iloc[:, 1].mean() for run_df in runs]  # success is the second column

	# Find indices of worst and best
	min_idx = avgs.index(min(avgs))
	max_idx = avgs.index(max(avgs))

	# Remove the worst and best runs
	if min_idx != max_idx:
		runs.pop(max(min_idx, max_idx))
		runs.pop(min(min_idx, max_idx))
	else:
		runs.pop(min_idx)

	# Merge all runs on steps
	merged = runs[0]
	#for df in runs[1:3] + runs[4:]:
	for df in runs[1:]:
		merged = pd.merge(merged, df, on="steps", how="inner")

	print(merged.head())
	success_cols = merged.columns.drop("steps")

	merged["mean"] = merged[success_cols].mean(axis=1)
	merged["mean"] = merged["mean"].rolling(3, min_periods=1).mean()
	merged["std"] = merged[success_cols].std(axis=1)
	merged["std"] = merged["std"].rolling(3, min_periods=1).mean()

	return merged[["steps", "mean", "std"]]


def main():
	plt.figure(figsize=(10, 6))

	algo_folders = sorted([
		f for f in os.listdir(ROOT_FOLDER)
		if os.path.isdir(os.path.join(ROOT_FOLDER, f))
	])

	algo_folders.remove("models")
	# algo_folders.remove("no_ds")
	algo_folders.remove("contraction_loss")
	# algo_folders.remove("no_kl")
	algo_folders.remove("no_kl_no_rew")
	algo_folders.remove("no_reward")
	algo_folders.remove("no_ds")
	algo_folders.remove("no_mask_no_double_step")
	for algo in algo_folders:
		print(algo, len(glob.glob(os.path.join(ROOT_FOLDER + '/' + algo, "res_*.csv"))))
		folder_path = os.path.join(ROOT_FOLDER, algo)

		df = process_algorithm(folder_path)

		# Plot mean
		plt.plot(df["steps"], df["mean"], label=algo)

		# Plot variance (shaded)
		plt.fill_between(
			df["steps"],
			df["mean"] - df["std"],
			df["mean"] + df["std"],
			alpha=0.2
		)

	plt.xlabel("Environment Steps")
	plt.ylabel("Success Rate")
	plt.title("Algorithm Comparison (Mean ± Std)")
	plt.legend()
	plt.grid(True)
	plt.ylim(0, 1)

	plt.tight_layout()
	plt.savefig('fig_all.png', dpi=400)


if __name__ == "__main__":
	main()