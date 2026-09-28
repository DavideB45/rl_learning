import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = 'calibration_readings.csv'

# Set to False to plot only the increasing/decreasing mean curves and the fit
# line. Useful when the individual runs and the hysteresis gap are not
# informative (e.g. no visible run-to-run spread or hysteresis).
SHOW_INDIVIDUAL_RUNS = False

# Same validated CVD-safe categorical colors used across the thesis plots
# (analysis/final_plot.py EXPERIMENT_COLORS palette).
COLOR_INCREASING = "#2a78d6"  # blue
COLOR_DECREASING = "#eb6834"  # orange
COLOR_FIT = "#3d3d3d"         # neutral dark gray, doesn't compete with the two hues

# Different marker shapes per direction so the two curves stay distinguishable
# even where they overlap almost exactly (color alone isn't enough there).
MARKER_INCREASING = 'o'
MARKER_DECREASING = 'x'

# ==========================================
# DATA
# ==========================================
df = pd.read_csv(CSV_PATH)
df_inc = df[df['direction'] == 'increasing']
df_dec = df[df['direction'] == 'decreasing']

# Fit a linear approximation, excluding raw_value=255: the valve is saturated
# there, so it isn't representative of the linear regime the fit is meant to
# describe (the python sender uses this formula to convert bar -> raw byte).
df_fit = df[df['raw_value'] != 255]
x_all = df_fit['raw_value'].to_numpy()
y_all = df_fit['user_reading'].to_numpy()
slope, intercept = np.polyfit(x_all, y_all, 1)

x_fit = np.linspace(x_all.min(), x_all.max(), 100)
approx_y = slope * x_fit + intercept

y_pred = slope * x_all + intercept
ss_res = np.sum((y_all - y_pred) ** 2)
ss_tot = np.sum((y_all - y_all.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot

# The CSV concatenates several increasing/decreasing calibration runs (raw=255
# appears multiple times per direction). Average across runs to get a clean
# mean curve per direction, and use the gap between the two mean curves to
# visualize the hysteresis loop.
mean_inc = df_inc.groupby('raw_value')['user_reading'].mean().reset_index().sort_values('raw_value')
mean_dec = df_dec.groupby('raw_value')['user_reading'].mean().reset_index().sort_values('raw_value')
hysteresis = pd.merge(mean_inc, mean_dec, on='raw_value', suffixes=('_inc', '_dec'))

# ==========================================
# PLOT
# ==========================================
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(9, 6))

if SHOW_INDIVIDUAL_RUNS:
    # Individual readings from each calibration run, shown faint underneath
    # the mean curves so run-to-run spread is visible without cluttering the
    # trend.
    ax.scatter(df_inc['raw_value'], df_inc['user_reading'],
               color=COLOR_INCREASING, alpha=0.35, s=35, marker=MARKER_INCREASING, edgecolor='none',
               label='Increasing (individual runs)', zorder=2)
    ax.scatter(df_dec['raw_value'], df_dec['user_reading'],
               color=COLOR_DECREASING, alpha=0.35, s=45, marker=MARKER_DECREASING,
               label='Decreasing (individual runs)', zorder=2)

    # Shaded hysteresis gap between the two directions' mean curves
    ax.fill_between(hysteresis['raw_value'], hysteresis['user_reading_inc'], hysteresis['user_reading_dec'],
                     color='gray', alpha=0.15, zorder=1, label='Hysteresis gap')

mean_inc_label = 'Increasing (mean)' if SHOW_INDIVIDUAL_RUNS else 'Increasing'
mean_dec_label = 'Decreasing (mean)' if SHOW_INDIVIDUAL_RUNS else 'Decreasing'

# Mean curve per direction
ax.plot(mean_inc['raw_value'], mean_inc['user_reading'],
         color=COLOR_INCREASING, linewidth=2.2, marker=MARKER_INCREASING, markersize=7,
         label=mean_inc_label, zorder=3)
ax.plot(mean_dec['raw_value'], mean_dec['user_reading'],
         color=COLOR_DECREASING, linewidth=2.2, marker=MARKER_DECREASING, markersize=9,
         markeredgewidth=2, label=mean_dec_label, zorder=4)

# Global line of best fit (always shown, drawn on top of everything else)
ax.plot(x_fit, approx_y,
         linestyle='--', color=COLOR_FIT, linewidth=2,
         label=f'Global fit: y = {slope:.4f}x + {intercept:.4f}  ($R^2$ = {r_squared:.3f})',
         zorder=5)

ax.set_xlabel('Raw Command Value (0–255)', fontsize=12)
ax.set_ylabel('Pressure Reading (bar)', fontsize=12)
ax.set_title('Pressure Valve Calibration — Hysteresis Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=300)
plt.show()
