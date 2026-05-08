import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==== CONFIG ====
"/data/drawer-open/full_experiments/contraction_loss"
CSV_PATTERN = "./res_*.csv"   # matches res_1.csv, res_2.csv, ...
EVAL_EPISODES = 5             # 5 test episodes per evaluation
EVAL_FREQUENCY = 5000         # steps between evaluations
# =================


def process_single_run(file_path):
    df = pd.read_csv(file_path)

    # Convert success to int
    df["success"] = df["success"].astype(int)

    # Group every 5 rows (one evaluation)
    df["eval_id"] = np.arange(len(df)) // EVAL_EPISODES

    grouped = df.groupby("eval_id")["success"].mean().reset_index()

    # Convert eval index → environment steps
    grouped["steps"] = grouped["eval_id"] * EVAL_FREQUENCY

    return grouped[["steps", "success"]]


def main():
    files = sorted(glob.glob(CSV_PATTERN))

    all_runs = []

    for f in files:
        run_df = process_single_run(f)
        run_df = run_df.rename(columns={"success": os.path.basename(f)})
        all_runs.append(run_df)

    # files = sorted(glob.glob("./data/drawer-open/full_experiments/committment_loss/res_*.csv"))
    # for f in files:
    #     run_df = process_single_run(f)
    #     run_df = run_df.rename(columns={"success": os.path.basename(f)})
    #     all_runs.append(run_df)

    # Merge all runs on "steps"
    merged = all_runs[0]
    for df in all_runs[1:]:
        merged = pd.merge(merged, df, on="steps", how="inner")

    # Extract only success columns
    success_cols = merged.columns.drop("steps")

    # Compute mean and std across runs
    merged["mean"] = merged[success_cols].mean(axis=1)
    merged["mean"] = merged["mean"].rolling(5, min_periods=1).mean()
    merged["std"] = merged[success_cols].var(axis=1)
    merged["std"] = merged["std"].rolling(5, min_periods=1).mean()

    # ==== PLOT ====
    plt.figure(figsize=(10, 6))

    plt.plot(merged["steps"], merged["mean"], label="Mean Success Rate")

    plt.fill_between(
        merged["steps"],
        merged["mean"] - merged["std"],
        merged["mean"] + merged["std"],
        alpha=0.3,
        label="±1 var"
    )

    plt.xlabel("Environment Steps")
    plt.ylabel("Success Rate")
    plt.title("Evaluation Performance (Mean ± Var)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('fig.png', dpi=400)


if __name__ == "__main__":
    main()