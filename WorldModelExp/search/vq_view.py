import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import matplotlib.pyplot as plt
import pandas as pd

from global_var import CURRENT_ENV

def load_all_histories(path):
	histories = {}
	for fname in os.listdir(path):
		if fname.endswith(".csv"):
		#if fname.endswith(".txt"):
			with open(os.path.join(path, fname), "r") as f:
				histories[fname[:-4]] = pd.read_csv(f)
	return histories

def get_best_model_keys(all:dict) -> list[tuple[float, str]]:
	sorted_keys = []
	for name, hist in all.items():
		#Val_total_loss, Val_recon_loss, Val_flatness_loss
		best = min(hist['Val_recon_loss'])
		sorted_keys.append((best, name))
	sorted_keys = sorted(sorted_keys, reverse=False)
	return sorted_keys

def plot_metric_across_runs(histories: dict, set="Train", metric="total_loss", bool_log=False):

	plt.figure(figsize=(7, 4))
	for name, hist in histories.items():
		plt.plot(hist[f"{set.capitalize()}_{metric}"], label=name)

	plt.title(f"{metric} across runs ({set} set)")
	plt.xlabel("Epoch")
	plt.ylabel(metric)
	plt.legend()
	#max_y = 6
	#plt.ylim(2, max_y)
	if bool_log:
		plt.yscale('log')
	plt.tight_layout()
	plt.savefig(f"images/fig_vq_{set}_{metric}_across_runs.png")


if __name__ == '__main__':


	h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/vq")
	#h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/vq/many")
	#h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/vq/no_smoothing/")
	sorted_keys = get_best_model_keys(h)
	print(" --- --- [MODELS FOUNDED] --- --- ")
	for i, name in enumerate(sorted_keys):
		print(f"{i}: {name}")

	print(" --- --- [BEST MODELS PER PREFIX] --- --- ")
	best_each = {}
	for best, name in sorted_keys:
		prefix = name.split('ema')[0]
		suffix = name.split('sm')[1]
		prefix = prefix + "sm" + suffix
		if prefix not in best_each or best_each[prefix][0] > best:
			best_each[prefix] = (best, name)
	for prefix, (best, name) in best_each.items():
		print(f"Model: {name} | Best Val Loss: {best}")

	to_plot = {name: h[name] for _, name in best_each.values()}

	# flatness_loss, recon_loss, total_loss
	plot_metric_across_runs(to_plot, metric="recon_loss", set="Val", bool_log=True)