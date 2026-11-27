import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_PER_POLICY = 10
SMOOTH_WINDOW = 3  # try 3 or 4

data = pd.read_csv("res_0.csv").iloc[:, 0]

# Average each block of 10 consecutive runs (one block = one policy)
per_policy = data.groupby(np.arange(len(data)) // RUNS_PER_POLICY).mean()

# Light smoothing across policies
smoothed = per_policy.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

EVAL_EVERY = 5100
steps = (per_policy.index + 1) * EVAL_EVERY

plt.plot(steps, per_policy, alpha=0.3, label="per-policy mean")
plt.plot(steps, smoothed, label=f"smoothed (window={SMOOTH_WINDOW})")
plt.ylim(-22, 4)
plt.xlabel("Training steps")
plt.ylabel("Mean result")
plt.legend()
plt.minorticks_on()
plt.grid(True, which="major", alpha=0.4)
plt.grid(True, which="minor", alpha=0.15, linestyle=":")
plt.gca().xaxis.set_major_formatter(lambda x, _: f"{x/1000:.0f}k")
plt.savefig("vq_rec_default.png")