import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==== CONFIG ====
ENV = 'peg-insert'
CSV_FOLDER = f"./data/{ENV}/full_experiments/reset_conditional/"   # folder containing your CSV files
ROLLING_WINDOW = 10*3     # number of episodes for rolling mean
INTERACTIONS_PER_EPISODE = 500

# =================

def load_and_process(file_path):
	df = pd.read_csv(file_path)

	if "MMWM" in os.path.basename(file_path):
		prefix = pd.DataFrame({"success": [False] * 20})
		df = pd.concat([prefix, df], ignore_index=True)

	# Convert success to int (True -> 1, False -> 0)
	df["success"] = df["success"].astype(int)

	# Episode index
	df["episode"] = range(len(df))

	# Convert to environment interactions
	df["interactions"] = df["episode"] * INTERACTIONS_PER_EPISODE

	# Rolling mean of success
	df["success_rate"] = df["success"].rolling(
		window=ROLLING_WINDOW, min_periods=1
	).mean()

	return df


def main():
	plt.figure(figsize=(10, 10))

	csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))

	dfs = []
	min_len = None

	for file_path in csv_files:
		df = load_and_process(file_path)
		label = os.path.basename(file_path).replace(".csv", "")
		dfs.append((df, label))

		if min_len is None or len(df) < min_len:
			min_len = len(df)

	for df, label in dfs:
		df = df.iloc[:min_len]
		max_success = df["success_rate"].max()
		print(f"{label}: max success_rate = {max_success:.4f}")
		plt.plot(
			df["interactions"],
			df["success_rate"],
			label=label,
			linewidth=4
		)

	plt.xlabel("Environment Interactions", fontsize=20)
	plt.ylabel("Success Rate (Rolling Mean)", fontsize=20)
	plt.title(f"{ENV} Performance (window={ROLLING_WINDOW})", fontsize=25)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(f'{ENV}.png', dpi=300)


if __name__ == "__main__":
	main()