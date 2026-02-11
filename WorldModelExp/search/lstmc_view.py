import json
import os
import sys
import matplotlib.pyplot as plt
import torch

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV
from helpers.model_loader import load_vq_vae
from helpers.general import best_device
from torch import cdist, triu

VAE_TO_TEST = [(4, 16, 64)] # latent, code_depth, codebook_size (4, 16, 128), 
LEARNING_RATES=[1e-5, 2e-5, 5e-5]
LAMBDA_REGS = [0, 5e-4, 1e-3]
USE_KL = [True]

HIDDEN_DIM = 1024
SEQ_LEN = 7
INIT_LEN = 2


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

	# CE
	axs[0].plot(history["tr"]["ce"], label="train")
	axs[0].plot(history["vl"]["ce"], label="val")
	axs[0].set_title("Cross-Entropy")
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
	
def plot_metric_across_runs(histories: dict, set="tr", metric="ce"):

	plt.figure(figsize=(7, 4))
	for name, hist in histories.items():
		plt.plot(hist[set][metric], label=name)#, color='r' if '128' in name.split('_') else 'g')

	plt.title(metric + ' ' + set)
	plt.xlabel("Epoch")
	plt.ylabel(metric)
	plt.legend(fontsize=8)
	plt.tight_layout()
	plt.savefig(f"images/all_{set}_{metric}")

def filter_parameter(all:dict, codebook_sizes=[128, 64], kl=[True, False], lr=[1e-5, 2e-5], wd=[0, 1e-3, 2e-3]) -> dict:
	interest_files = {}
	for name, hist in all.items():
		if isinstance(name, str):
			model, size, ld, cd, cs, klc, lrc, wdc, fl = tuple(name.split("_"))
			cs = int(cs)
			klc= True if klc == "True" else False
			lrc=float(lrc)
			wdc=float(wdc)
			if cs in codebook_sizes and klc in kl and lrc in lr and wdc in wd:
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
		model, size, ld, cd, cs, kl, lr, wdc, fl = tuple(name.split("_"))
		prefix = f'{size}_{ld}_{cd}_{cs}_{kl}_{fl}'
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

	LEARNING_RATE=1e-5
	LAMBDA_REG = 0
	USE_KL = False

	LATENT_DIM = 4
	CODE_DEPTH = 16
	CDODEBOOK_SIZE = 64
	FLATTENED = True

	vq_f = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, True, best_device())
	vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, False, best_device())
	pairwise_distances = cdist(vq_f.quantizer.embedding.weight, vq_f.quantizer.embedding.weight, p=2)
	upper_triangle_mask = triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
	distances = pairwise_distances[upper_triangle_mask]
	avg_distance = distances.mean()
	print('Quantizer with flat loss - Average pairwise distance:', avg_distance.item())
	
	pairwise_distances = cdist(vq.quantizer.embedding.weight, vq.quantizer.embedding.weight, p=2)
	upper_triangle_mask = triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
	distances = pairwise_distances[upper_triangle_mask]
	avg_distance = distances.mean()
	print('Quantizer without flat loss - Average pairwise distance:', avg_distance.item())

	file_name = f'lstmc_{HIDDEN_DIM}_{LATENT_DIM}_{CODE_DEPTH}_{CDODEBOOK_SIZE}_{USE_KL}_{LEARNING_RATE}_{LAMBDA_REG}_{FLATTENED}'
	
	h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/2318_smooth/")
	sorted_keys = get_best_model_keys(h)
	print(" --- --- [MODELS FOUNDED] --- --- ")
	for i, name in enumerate(sorted_keys):
		print(f"{i}: {name}")
	get_best_for_each(h, metric='mse')
	bests = get_best_for_each(h, metric='acc')
	to_plot = {}
	for pr in bests:
		to_plot[bests[pr][1]] = h[bests[pr][1]]
	plot_history(h[file_name], title=file_name)
	h = filter_parameter(h, kl=[False], codebook_sizes=[64])
	plot_metric_across_runs(to_plot, metric="mse", set="vl")

