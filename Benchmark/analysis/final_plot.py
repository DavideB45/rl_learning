import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
#ENV_NAME = "button-press"
#ENV_NAME = "button-press-td"
#ENV_NAME = "drawer-open"
#ENV_NAME = "window-open"
ENV_NAME = "peg-insert"
BASE_DIR = f"data/{ENV_NAME}/full_experiments/"
TITLE = "proprioception"
#BASE_DIR = f"data/{ENV_NAME}/stabilization/"

# Comment out the ones you DON'T want to plot
EXPERIMENTS_TO_PLOT = [
    #"Dreamer_160",
    #"Dreamer_23",
    #"default",
    #"no_kl",
    #"no_mask2",
    #"propioception",
    "ppo_default",
    "ppo_impala",
    #"teacher_forcing",
    #"propioception_0.01",
    #"reset_500",
    #"reset_conditional",
    #"reset_conditional_restore_prop"
]

# Evaluation parameters
EPISODES_PER_EVAL = 10
EPISODE_LENGTH = 500
STEPS_PER_EVAL = EPISODE_LENGTH * EPISODES_PER_EVAL # 5000 steps

# Rolling window size (number of evaluation points to smooth over)
ROLLING_WINDOW = 3

# Number of steps spent on initial (untrained) data collection before training
# actually starts. Dreamer and PPO runs start training immediately, but every
# other model first gathers this many steps of data, so their x-axis needs to
# be shifted right by this amount to reflect real elapsed environment steps.
WARMUP_STEPS = 10000
NO_WARMUP_PREFIXES = ("dreamer", "ppo")  # case-insensitive exp_name prefixes

# ==========================================
# COLOR & LABEL REGISTRY (fixed across the whole thesis)
# ==========================================
# Every experiment folder name that ever gets plotted should have an entry
# here. Colors are assigned ONCE, globally, so the same experiment always
# gets the same color no matter which subset/plot it appears in. Colors 1-8
# come from a validated CVD-safe categorical palette; the rest are a
# secondary set for the less-frequently-combined ablations.
#   Anchor:        "default" is green in EVERY plot - it's the one series that
#                  is always present, so it acts as the visual reference point.
#   Main comparison (default + both Dreamer sizes + PPO, 4 lines together):
#                  the two Dreamer runs share the blue family (dark vs light,
#                  same hue so they read as "the same model, different size"),
#                  PPO variants share the orange/yellow family. Green/blue/
#                  orange/yellow are four hues that are easy to tell apart at
#                  a glance and none of them are reused by the ablations below.
#   Regular ablations (each only ever plotted 2-lines-at-a-time against
#                  "default", never against each other): they all intentionally
#                  SHARE one color (red) since they never co-occur in a figure.
#   Proprioception ablations: these are called out as the more important
#                  comparison, so they get their own violet family instead of
#                  the shared ablation red - it visually stands out as "the
#                  interesting one" wherever it appears. The three variants
#                  are shades of the same violet in case two of them are ever
#                  plotted together.
EXPERIMENT_COLORS = {
    "default":                        "#008300",  # green  - anchor, always "Ours"

    "Dreamer_160":                     "#104281",  # dark navy blue
    "Dreamer_23":                      "#4e75a4",  # light sky blue (same family as 160)
    "ppo_default":                     "#eb6834",  # orange
    "ppo_impala":                      "#eda100",  # yellow (same family as ppo_default)

    "no_kl":                           "#e34948",  # red - shared "regular ablation" color
    "no_mask":                         "#e34948",  # red - shared "regular ablation" color
    "no_mask2":                        "#e34948",  # red - shared "regular ablation" color
    "teacher_forcing":                 "#e34948",  # red - shared "regular ablation" color
    "reset_500":                       "#e34948",  # red - shared "regular ablation" color
    "reset_conditional":               "#e34948",  # red - shared "regular ablation" color
    "reset_conditional_restore_prop":  "#e34948",  # red - shared "regular ablation" color

    "propioception":                   "#4a3aa7",  # violet - the important comparison
    "propioception_0.01":              "#9085e9",  # lighter violet, same family
    "prop_rr":                         "#7a5fc4",  # mid violet, same family
}

# Human-readable legend/table labels. These are best-effort guesses based on
# folder names -- EDIT to match the terminology used in the thesis text.
EXPERIMENT_LABELS = {
    "default":                        "Ours",
    "Dreamer_160":                     "DreamerV3 (160M)",
    "Dreamer_23":                      "DreamerV3 (23M)",
    "ppo_default":                     "PPO (NatureCNN)",
    "ppo_impala":                      "PPO (IMPALA)",
    "no_kl":                           "Ours (no KL loss)",
    "no_mask":                         "Ours (no mask)",
    "no_mask2":                        "Ours (no mask)",
    "propioception":                   "Ours (Multimodal)",
    "propioception_0.01":              "Ours (proprioception, 0.01)",
    "prop_rr":                         "Ours (proprioception, reduced rate)",
    "teacher_forcing":                 "Ours (teacher forcing)",
    "reset_500":                       "Ours (reset @ 500)",
    "reset_conditional":               "Ours (conditional reset)",
    "reset_conditional_restore_prop":  "Ours (conditional reset, restore prop.)",
}

# Overflow palette for any experiment name not registered above, so the
# script never crashes -- but a warning is printed since that color/label
# is NOT guaranteed consistent across other plots.
_FALLBACK_PALETTE = sns.color_palette("husl", 12).as_hex()
_fallback_cache = {}

def get_color(exp_name):
    if exp_name in EXPERIMENT_COLORS:
        return EXPERIMENT_COLORS[exp_name]
    if exp_name not in _fallback_cache:
        print(f"WARNING: '{exp_name}' has no registered color. "
              f"Add it to EXPERIMENT_COLORS for thesis-wide consistency.")
        _fallback_cache[exp_name] = _FALLBACK_PALETTE[len(_fallback_cache) % len(_FALLBACK_PALETTE)]
    return _fallback_cache[exp_name]

def get_label(exp_name):
    if exp_name not in EXPERIMENT_LABELS:
        print(f"WARNING: '{exp_name}' has no registered label. "
              f"Add it to EXPERIMENT_LABELS. Falling back to the folder name.")
    return EXPERIMENT_LABELS.get(exp_name, exp_name)

def needs_warmup_shift(exp_name):
    return not exp_name.lower().startswith(NO_WARMUP_PREFIXES)

# ==========================================
# DATA PROCESSING
# ==========================================
def load_and_process_data():
    all_data = []
    base_path = Path(BASE_DIR)

    for exp_name in EXPERIMENTS_TO_PLOT:
        exp_path = base_path / exp_name

        if not exp_path.exists():
            print(f"Warning: Directory {exp_path} not found. Skipping.")
            continue

        for file_path in exp_path.glob("*.csv"):
            run_id = file_path.stem

            df = pd.read_csv(file_path)

            df['success'] = df['success'].astype(float)

            chunked_df = df.groupby(df.index // EPISODES_PER_EVAL).mean()
            chunked_df['step'] = chunked_df.index * STEPS_PER_EVAL

            if needs_warmup_shift(exp_name):
                chunked_df['step'] += WARMUP_STEPS

            chunked_df['mrew_smooth'] = chunked_df['mrew'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            chunked_df['success_smooth'] = chunked_df['success'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

            chunked_df['experiment'] = exp_name
            chunked_df['label'] = get_label(exp_name)
            chunked_df['run'] = run_id

            if needs_warmup_shift(exp_name):
                # Anchor the line at (0, 0) so it visibly rises from the
                # bottom during the warmup gathering phase, instead of
                # popping in mid-air at step=WARMUP_STEPS with whatever
                # value the first real evaluation happened to have.
                anchor = pd.DataFrame([{
                    'step': 0,
                    'mrew': 0.0,
                    'success': 0.0,
                    'mrew_smooth': 0.0,
                    'success_smooth': 0.0,
                    'experiment': exp_name,
                    'label': get_label(exp_name),
                    'run': run_id,
                }])
                chunked_df = pd.concat([anchor, chunked_df], ignore_index=True)

            all_data.append(chunked_df)

    df_all = pd.concat(all_data, ignore_index=True)

    # --- TRUNCATION LOGIC ---
    # Find the maximum step reached by each individual run
    max_steps_per_run = df_all.groupby(['experiment', 'run'])['step'].max()

    # The cutoff is the shortest of those maximums
    cutoff_step = max_steps_per_run.min()

    print(f"Shortest run ends at step {cutoff_step}. Truncating all data to match.")

    # Filter out anything past the cutoff
    df_all = df_all[df_all['step'] <= cutoff_step]
    # ----------------------------

    return df_all

# ==========================================
# PLOTTING
# ==========================================
def plot_results(data):
    sns.set_theme(style="darkgrid")

    hue_order = [get_label(e) for e in EXPERIMENTS_TO_PLOT if e in data['experiment'].unique()]
    palette = {get_label(e): get_color(e) for e in EXPERIMENTS_TO_PLOT}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean Reward
    sns.lineplot(
        data=data,
        x='step',
        y='mrew_smooth',
        hue='label',
        hue_order=hue_order,
        palette=palette,
        errorbar=('ci', 95),
        ax=axes[0]
    )
    axes[0].set_title('Mean Reward (Bootstrapped CI)')
    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Reward')

    # Plot 2: Success Rate
    sns.lineplot(
        data=data,
        x='step',
        y='success_smooth',
        hue='label',
        hue_order=hue_order,
        palette=palette,
        errorbar=('ci', 95),
        ax=axes[1]
    )
    axes[1].set_title('Success Rate (Bootstrapped CI)')
    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f'final_plot_{TITLE}_{ENV_NAME}.png', dpi=300)
    #plt.show()

# ==========================================
# FINAL SUCCESS RATE SUMMARY (for thesis tables)
# ==========================================
def bootstrap_ci(values, n_boot=10000, ci=95, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = np.random.default_rng(seed)
    resampled_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = np.percentile(resampled_means, (100 - ci) / 2)
    upper = np.percentile(resampled_means, 100 - (100 - ci) / 2)
    return values.mean(), lower, upper

def print_final_success_summary(data):
    print("\n" + "=" * 70)
    print(f"FINAL SUCCESS RATE SUMMARY -- {ENV_NAME} ({TITLE})")
    print("=" * 70)

    final_step = data['step'].max()
    rows = []
    for exp_name in EXPERIMENTS_TO_PLOT:
        exp_data = data[data['experiment'] == exp_name]
        if exp_data.empty:
            continue

        # For each run, take its value at the final (shared) step.
        per_run_final = (
            exp_data[exp_data['step'] == final_step]
            .groupby('run')['success_smooth']
            .mean()
        )

        mean, lo, hi = bootstrap_ci(per_run_final.values)
        rows.append({
            'experiment': exp_name,
            'label': get_label(exp_name),
            'n_runs': len(per_run_final),
            'mean': mean,
            'ci_lo': lo,
            'ci_hi': hi,
        })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        print("No data to summarize.")
        return summary_df

    print(f"(evaluated at step={final_step}, n_boot=10000, 95% CI)\n")
    for _, r in summary_df.iterrows():
        print(f"  {r['label']:<45} {r['mean']*100:6.2f}%  "
              f"95% CI [{r['ci_lo']*100:.2f}, {r['ci_hi']*100:.2f}]  (n={r['n_runs']})")

    print("\n-- LaTeX table rows --")
    for _, r in summary_df.iterrows():
        half_width = (r['ci_hi'] - r['ci_lo']) / 2
        print(f"{r['label']} & {r['mean']*100:.1f} $\\pm$ {half_width*100:.1f} \\\\")

    return summary_df

if __name__ == "__main__":
    print("Processing data...")
    df_all = load_and_process_data()
    print("Generating plots...")
    plot_results(df_all)
    print_final_success_summary(df_all)
