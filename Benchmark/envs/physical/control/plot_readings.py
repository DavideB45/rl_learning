import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('calibration_readings.csv')

# Split the data into two dataframes based on the direction flag
df = df[df['raw_value'] != 255]
df_inc = df[df['direction'] == 'increasing']
df_dec = df[df['direction'] == 'decreasing']


# Fit a linear approximation using ALL data points 
# (This gives you the best average formula for your python sender)
x_all = df['raw_value'].to_numpy()
y_all = df['user_reading'].to_numpy()
slope, intercept = np.polyfit(x_all, y_all, 1)

# Generate a clean, straight line of X values for the fit line plot
x_fit = np.linspace(x_all.min(), x_all.max(), 100)
approx_y = slope * x_fit + intercept

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the increasing and decreasing paths as points (no connecting lines)
plt.plot(df_inc['raw_value'], df_inc['user_reading'], 
         marker='o', linestyle='None', markersize=6, 
         label='Increasing', alpha=0.9)

plt.plot(df_dec['raw_value'], df_dec['user_reading'], 
         marker='x', linestyle='None', markersize=6, 
         label='Decreasing', alpha=0.9)

# Plot the global line of best fit in black dashes
plt.plot(x_fit, approx_y, 
         linestyle='--', color='black', linewidth=2, 
         label=f'Global Fit: y = {slope:.4f}x + {intercept:.4f}')

plt.xlabel('Raw Value (0-255)')
plt.ylabel('Pressure Reading (bar)')
plt.title('Pressure Valve Calibration (Hysteresis Curve)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig('calibration_plot.png')
plt.show()