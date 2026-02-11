import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV

VAE_TO_TEST = [(4, 16, 128), (4, 16, 64)] # latent, code_depth, codebook_size
NUM_EPOCS=100 # this is (if there is no early stopping around 1 our per model)
LEARNING_RATES=[1e-5, 2e-5, 5e-5]
LAMBDA_REGS = [0, 5e-4, 1e-3]

HIDDEN_DIM = 1024
SEQ_LEN = 23
INIT_LEN = 18


def load_history(path, version):
	with open(os.path.join(path, f"{version}.json"), "r") as f:
		return json.load(f)

def load_all_histories(path):
	histories = {}
	for fname in os.listdir(path):
		if fname.endswith(".json"):
			with open(os.path.join(path, fname), "r") as f:
				histories[fname[:-5]] = json.load(f)
	return histories

def plot_history(history: dict, title=None):
	
	fig, axs = plt.subplots(1, 3, figsize=(15, 4))

	# MSEQ
	axs[0].plot(history["tr"]["qmse"], label="train")
	axs[0].plot(history["vl"]["qmse"], label="val")
	axs[0].set_title("MSE-q")
	axs[0].legend()

	# MSE
	axs[1].plot(history["tr"]["mse"], label="train")
	axs[1].plot(history["vl"]["mse"], label="val")
	axs[1].set_title("MSE")
	axs[1].legend()

	# Accuracy
	axs[2].plot(history["tr"]["acc"], label="train")
	axs[2].plot(history["vl"]["acc"], label="val")
	axs[2].set_title("Accuracy")
	axs[2].legend()

	if title:
		fig.suptitle(title)

	plt.tight_layout()
	plt.savefig(f"images/fig_{title}.png")
	
def plot_metric_across_runs(histories: dict, set="tr", metric="mseq"):

	plt.figure(figsize=(7, 4))
	for name, hist in histories.items():
		plt.plot(hist[set][metric], label=name, color='r' if 'False' in name.split('_') else 'g')

	plt.title(metric + ' ' + set)
	plt.xlabel("Epoch")
	plt.ylabel(metric)
	plt.legend(fontsize=8)
	plt.tight_layout()
	plt.savefig(f"images/all_{set}_{metric}")

def filter_parameter(all:dict, codebook_sizes=[128, 64], lr=LEARNING_RATES, wd=LAMBDA_REGS) -> dict:
	interest_files = {}
	for name, hist in all.items():
		if isinstance(name, str):
			model, size, ld, cd, cs, lrc, wdc, fl = tuple(name.split("_"))
			cs = int(cs)
			lrc=float(lrc)
			wdc=float(wdc)
			if cs in codebook_sizes and lrc in lr and wdc in wd:
				interest_files[name] = hist
		else:
			raise RuntimeError("Keys must be strings")
	return interest_files

def get_best_model_keys(all:dict) -> list[tuple[float, str]]:
	sorted_keys = []
	for name, hist in all.items():
		best = max(hist['vl']['acc'])
		sorted_keys.append((best, name))
	sorted_keys = sorted(sorted_keys, reverse=True)
	return sorted_keys

def get_best_for_each(all:dict, metric:str) -> list[tuple[float, str]]:
	best_each = {}
	for name, hist in all.items():
		model, size, ld, cd, cs, lr, wdc, fl = tuple(name.split("_"))
		prefix = f'{size}_{ld}_{cd}_{cs}_{fl}'
		if metric == 'mse':
			best = min(hist['vl'][metric])
			if prefix not in best_each or best_each[prefix][0] > best:
				best_each[prefix] = (best, name)
		elif metric =='acc':
			best = max(hist['vl'][metric])
			if prefix not in best_each or best_each[prefix][0] < best:
				best_each[prefix] = (best, name)
	print(f" --- --- [BEST MODELS ({metric})] --- --- ")
	for prefix, (best, name) in best_each.items():
		print(f"Model: {name} | Best Val Loss: {best}")
	return best_each


if __name__ == '__main__':

	LEARNING_RATE=5e-5
	LAMBDA_REG = 0.001

	LATENT_DIM = 4
	CODE_DEPTH = 16
	CDODEBOOK_SIZE = 64
	FLATTEN = True

	file_name = f'lstm_{HIDDEN_DIM}_{LATENT_DIM}_{CODE_DEPTH}_{CDODEBOOK_SIZE}_{LEARNING_RATE}_{LAMBDA_REG}_{FLATTEN}'

	h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/2318_smooth_reg/")
	sorted_keys = get_best_model_keys(h)
	print(" --- --- [MODELS FOUNDED] --- --- ")
	for i, name in enumerate(sorted_keys):
		print(f"{i}: {name}")

	get_best_for_each(h, metric="acc")
	get_best_for_each(h, metric="mse")
	plot_history(h[file_name], title=file_name)
	h = filter_parameter(h, codebook_sizes=[64])
	plot_metric_across_runs(h, metric="acc", set="vl")

