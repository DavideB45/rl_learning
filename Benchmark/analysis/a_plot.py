import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

#data=[float(x) for x in subprocess.check_output("grep , res.csv | tail n +1 |awk -F',' '{print $1}'", shell=True).decode().split()]
data = [float(x) for x in subprocess.check_output(
    "tail -n +2 data/drawer-open/numbers/4_norm_lrdec_noinit/res.csv | cut -d',' -f3",
    shell=True
).decode().split()]
# data = [str(x) for x in subprocess.check_output(
#     "tail -n +2 res.csv | cut -d',' -f2",
#     shell=True
# ).split()]
#data = [1 if x == "b'True'" else 0 for x in data]
# data = [float(x.split('%')[0]) for x in subprocess.check_output(
#     'grep -e "Accurac" data/drawer-open/numbers/4_norm_lrdec_noinit/log3.out | cut -d "|" -f4',
#     shell=True
# ).decode().split()]
data = [float(x) for x in subprocess.check_output(
    'grep -e "Rew" data/drawer-open/numbers/4_norm_lrdec_noinit/log3.out | cut -d "|" -f4',
    shell=True
).decode().split()]
# data = [float(x.split('\x1b[0m')[0]) for x in subprocess.check_output(
#     'grep -e "recon_loss" data/drawer-open/numbers/4_norm_lrdec_noinit/log3.out | cut -d ":" -f3',
#     shell=True
# ).decode().split()]
# data = [float(x.split('\x1b[96m')[0].split('\x1b[0m')[0]) for x in subprocess.check_output(
#     'grep -e "flat" data/drawer-open/numbers/4_norm_lrdec_noinit/log3.out | cut -d ":" -f3',
#     shell=True
# ).decode().split()]
print(data)
# Convert to pandas Series
min_ = 0
max_ = 2
series = pd.Series(data)

# Compute rolling mean (window size = 3)
rolling_mean = series.rolling(window=10).mean()

# Plot only the rolling mean
plt.plot(rolling_mean)
plt.ylim(min_, max_)
plt.title("Rolling Mean")
plt.xlabel("Index")
plt.ylabel("Rolling Mean")
plt.savefig('vq_rec_default.png')