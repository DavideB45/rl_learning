"""
Appendix plot for a single, un-replicated run (e.g. the one-off SAC test).

Every other plot in the thesis compares MULTIPLE runs of the same
experiment, so its confidence band comes from bootstrapping across seeds
(see final_plot.py). With only one run there is nothing to bootstrap across
-- but each evaluation point is still an average over EPISODES_PER_EVAL
episodes, so we get a meaningful band "for free" from the sampling
uncertainty WITHIN that batch of episodes:

  - Success rate is a proportion (k successes out of n episodes) -> a
    Wilson score interval, the standard CI for a binomial proportion.
  - Mean reward is the mean of n samples -> a Student-t interval built
    from the sample std of those episodes' rewards.

To keep the noisy single-run curve readable, points are smoothed by
pooling ROLLING_WINDOW consecutive evaluation batches together (same idea
as the rolling mean in final_plot.py) -- which also pools their episodes,
giving a tighter, more honest interval as more evaluations accumulate.

Reuses the color/label registry and visual style from final_plot.py so
this figure matches every other plot in the thesis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from final_plot import get_color, get_label

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = "res_2.csv"
EXPERIMENT_KEY = "SAC"  # must exist in EXPERIMENT_COLORS / EXPERIMENT_LABELS in final_plot.py
ENV_NAME = "peg-insert"  # ASSUMPTION: set this to whichever env this SAC test actually used

# Evaluation parameters -- same convention as final_plot.py
EPISODES_PER_EVAL = 1
EPISODE_LENGTH = 100
STEPS_PER_EVAL = EPISODE_LENGTH * EPISODES_PER_EVAL

# Number of consecutive evaluation batches pooled together for smoothing.
# Pooling also pools their episodes, so the CI narrows as this grows.
ROLLING_WINDOW = 5

# SAC is a standard baseline algorithm, not "our" model, so ASSUMPTION: it
# has no initial warmup data-gathering phase -- unlike "our" model, it
# starts training from step 0 like Dreamer/PPO. Flip this if that's wrong.
HAS_WARMUP_SHIFT = False
WARMUP_STEPS = 10000

CI = 95

# ==========================================
# DATA PROCESSING
# ==========================================
def load_and_process_data():
    df = pd.read_csv(CSV_PATH)
    df['success'] = df['success'].astype(float)

    window_episodes = EPISODES_PER_EVAL * ROLLING_WINDOW
    min_episodes = EPISODES_PER_EVAL  # need at least one full eval batch to plot a point

    roll_mean_rew = df['mrew'].rolling(window=window_episodes, min_periods=min_episodes).mean()
    roll_std_rew = df['mrew'].rolling(window=window_episodes, min_periods=min_episodes).std(ddof=1)
    roll_n_rew = df['mrew'].rolling(window=window_episodes, min_periods=min_episodes).count()

    roll_success_mean = df['success'].rolling(window=window_episodes, min_periods=min_episodes).mean()
    roll_success_n = df['success'].rolling(window=window_episodes, min_periods=min_episodes).count()

    # Keep one row per completed evaluation batch (the last episode of each
    # batch of EPISODES_PER_EVAL), so the x-axis matches every other plot.
    chunk_rows = df.index[EPISODES_PER_EVAL - 1::EPISODES_PER_EVAL]
    chunk_rows = chunk_rows[roll_mean_rew.loc[chunk_rows].notna()]

    result = pd.DataFrame({
        'step': (chunk_rows // EPISODES_PER_EVAL) * STEPS_PER_EVAL,
        'mrew_smooth': roll_mean_rew.loc[chunk_rows].values,
        'rew_std': roll_std_rew.loc[chunk_rows].values,
        'rew_n': roll_n_rew.loc[chunk_rows].values,
        'success_smooth': roll_success_mean.loc[chunk_rows].values,
        'success_n': roll_success_n.loc[chunk_rows].values,
    })

    if HAS_WARMUP_SHIFT:
        result['step'] += WARMUP_STEPS
        anchor = pd.DataFrame([{
            'step': 0, 'mrew_smooth': 0.0, 'rew_std': 0.0, 'rew_n': 1,
            'success_smooth': 0.0, 'success_n': 1,
        }])
        result = pd.concat([anchor, result], ignore_index=True)

    # --- Confidence intervals ---
    # Mean reward: Student-t interval from the pooled episodes' std.
    t_crit = stats.t.ppf(1 - (1 - CI / 100) / 2, df=(result['rew_n'] - 1).clip(lower=1))
    rew_halfwidth = t_crit * result['rew_std'] / np.sqrt(result['rew_n'])
    result['mrew_lo'] = result['mrew_smooth'] - rew_halfwidth
    result['mrew_hi'] = result['mrew_smooth'] + rew_halfwidth

    # Success rate: Wilson score interval for a binomial proportion.
    z = stats.norm.ppf(1 - (1 - CI / 100) / 2)
    p, n = result['success_smooth'], result['success_n']
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    result['success_lo'] = ((center - halfwidth) / denom).clip(lower=0)
    result['success_hi'] = ((center + halfwidth) / denom).clip(upper=1)

    return result

# ==========================================
# PLOTTING
# ==========================================
def plot_results(data):
    sns.set_theme(style="darkgrid")

    color = get_color(EXPERIMENT_KEY)
    label = get_label(EXPERIMENT_KEY)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(data['step'], data['mrew_smooth'], color=color, label=label)
    axes[0].fill_between(data['step'], data['mrew_lo'], data['mrew_hi'], color=color, alpha=0.2)
    axes[0].set_title(f'Mean Reward ({CI}% CI)')
    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Reward')
    axes[0].legend()

    axes[1].plot(data['step'], data['success_smooth'], color=color, label=label)
    axes[1].fill_between(data['step'], data['success_lo'], data['success_hi'], color=color, alpha=0.2)
    axes[1].set_title(f'Success Rate ({CI}% CI)')
    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'final_plot_{EXPERIMENT_KEY.lower()}_{ENV_NAME}.png', dpi=300)
    plt.show()

def print_final_summary(data):
    last = data.iloc[-1]
    print("\n" + "=" * 70)
    print(f"FINAL SUCCESS RATE -- {get_label(EXPERIMENT_KEY)} on {ENV_NAME} "
          f"(single run, step={int(last['step'])})")
    print("=" * 70)
    print(f"  {last['success_smooth']*100:.2f}%  "
          f"{CI}% CI [{last['success_lo']*100:.2f}, {last['success_hi']*100:.2f}]  "
          f"(pooled n={int(last['success_n'])} episodes)")
    half_width = (last['success_hi'] - last['success_lo']) / 2 * 100
    print(f"\n-- LaTeX table row --\n{get_label(EXPERIMENT_KEY)} & "
          f"{last['success_smooth']*100:.1f} $\\pm$ {half_width:.1f} \\\\")

if __name__ == "__main__":
    print("Processing data...")
    data = load_and_process_data()
    print("Generating plot...")
    plot_results(data)
    print_final_summary(data)
