import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sys

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH      = sys.argv[1] if len(sys.argv) > 1 else "res.csv"
ROWS_PER_STEP = 2        # consecutive rows belonging to the same timestep
WINDOW        = 20       # rolling-average window (in timesteps)
SMOOTH_STD    = True     # shade ±1 std of rolling window
FIGSIZE       = (10, 5)
# ────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

# Normalise column names
df.columns = df.columns.str.strip().str.lower()
df["success"] = df["success"].astype(str).str.strip().str.lower() == "true"

# Group every ROWS_PER_STEP consecutive rows into one timestep
df["timestep"] = np.arange(len(df)) // ROWS_PER_STEP
grouped = df.groupby("timestep")["success"].agg(
    successes="sum",
    trials="count"
)
grouped["rate"] = grouped["successes"] / grouped["trials"]
grouped.index = np.arange(1, len(grouped) + 1)  # 1-based timestep index

# Rolling success rate over timesteps
roll_succ  = grouped["successes"].rolling(window=WINDOW, min_periods=1).sum()
roll_trial = grouped["trials"].rolling(window=WINDOW, min_periods=1).sum()
rate = roll_succ / roll_trial

# Std via binomial approximation: sqrt(p*(1-p)/n)
std = np.sqrt(rate * (1 - rate) / roll_trial).fillna(0)

fig, ax = plt.subplots(figsize=FIGSIZE)

# Shaded std band
if SMOOTH_STD:
    ax.fill_between(grouped.index,
                    np.clip(rate - std, 0, 1),
                    np.clip(rate + std, 0, 1),
                    alpha=0.18, color="#4C72B0", label=f"±1 std (window={WINDOW})")

# Rolling mean line
ax.plot(grouped.index, rate,
        color="#4C72B0", linewidth=2, label=f"Rolling success rate (window={WINDOW})")

# Raw scatter per timestep: 0, 0.5, or 1 depending on how many succeeded
ax.scatter(grouped.index, grouped["rate"],
           s=6, alpha=0.3, color="#3d3d2b", zorder=1,
           label="Per-timestep rate (0, 0.5, or 1)")

# Formatting
ax.set_xlabel("Environment Steps", fontsize=12)
ax.set_ylabel("Success rate", fontsize=12)
ax.set_title("Success Rate over Training", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(ticks=np.arange(-10, len(grouped)-10, 100), labels=np.arange(0, len(grouped), 100)/100)

plt.tight_layout()
out_path = './images/' + CSV_PATH.rsplit(".", 1)[0] + "_success_rate.png"
plt.savefig(out_path, dpi=400)
print(f"Saved → {out_path}")
plt.show()