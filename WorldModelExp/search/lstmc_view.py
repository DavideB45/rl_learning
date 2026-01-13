import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_var import CURRENT_ENV


VAE_TO_TEST = [(4, 16, 128), (4, 16, 64)] # latent, code_depth, codebook_size
NUM_EPOCS=65 # this is (if there is no early stopping around 1 our per model)
LEARNING_RATES=[1e-5, 2e-5]
LAMBDA_REGS = [0, 1e-3, 2e-3]
USE_KL = [True, False]

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
		plt.plot(hist[set][metric], label=name, color='r' if '128' in name.split('_') else 'g')

	plt.title(metric)
	plt.xlabel("Epoch")
	plt.ylabel(metric)
	plt.legend(fontsize=8)
	plt.tight_layout()
	plt.savefig(f"images/all_{set}_{metric}")

def filter_parameter(all:dict, codebook_sizes=[128, 64], kl=[True, False], lr=[1e-5, 2e-5], wd=[0, 1e-3, 2e-3]) -> dict:
	interest_files = {}
	for name, hist in all.items():
		if isinstance(name, str):
			model, size, ld, cd, cs, klc, lrc, wdc = tuple(name.split("_"))
			cs = int(cs)
			klc= True if klc == "True" else False
			lrc=float(lrc)
			wdc=float(wdc)
			if cs in codebook_sizes and klc in kl and lrc in lr and wdc in wd:
				interest_files[name] = hist
		else:
			raise RuntimeError("Keys must be strings")
	return interest_files

if __name__ == '__main__':

	LEARNING_RATE=1e-5
	LAMBDA_REG = 2e-3
	USE_KL = True

	LATENT_DIM = 4
	CODE_DEPTH = 16
	CDODEBOOK_SIZE = 64

	file_name = f'lstmc_{HIDDEN_DIM}_{LATENT_DIM}_{CODE_DEPTH}_{CDODEBOOK_SIZE}_{USE_KL}_{LEARNING_RATE}_{LAMBDA_REG}'
	
	h = load_all_histories(CURRENT_ENV['data_dir'] + "histories/")
	print(" --- --- [MODELS FOUNDED] --- --- ")
	for name in list(h.keys()):
		print(name)
	plot_history(h[file_name], title=file_name)
	h = filter_parameter(h, kl=[True])
	plot_metric_across_runs(h, metric="acc", set="vl")
