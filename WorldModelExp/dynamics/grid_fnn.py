import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.fnn import FNN
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_fnn
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import no_grad
import json
from itertools import product
from datetime import datetime

GRID = {
	"history_len": [1, 3, 5],
	"lr": [1e-5, 2e-5, 5e-5],
	"wd": [0.0, 1e-8, 1e-6],
}

RESULTS_FILE = "grid_search_results_fnn.json"

def train_model(history_len, lr, wd, epochs=100) -> tuple[float, float]:
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 128, 4, 4, True, dev)
	dyn_fnn = FNN(vq, dev, CURRENT_ENV['a_size'], history_len)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, history_len, 0.2, 64, max_ep=10000)

	optim = Adam(dyn_fnn.parameters(), lr=lr, weight_decay=wd)
	best_tr_qmse = float("inf")
	best_vl_qmse = float("inf")
	for i in range(epochs):
		err_tr = dyn_fnn.train_epoch(tr, optim, False)
		err_vl = dyn_fnn.eval_epoch(vl, False)

		best_tr_qmse = min(best_tr_qmse, err_tr["qmse"])
		best_vl_qmse = min(best_vl_qmse, err_vl["qmse"])

		print(
			f"{i:03d} | "
			f"tr qmse: {err_tr['qmse']:.5f} | "
			f"vl qmse: {err_vl['qmse']:.5f}",
			end='\b\r'
		)

	return best_tr_qmse, best_vl_qmse

def run_grid_search():
	results = []
	total_runs = ( len(GRID["history_len"]) * len(GRID["lr"]) * len(GRID["wd"]) )
	run_id = 0

	for h, lr, wd in product(GRID["history_len"], GRID["lr"], GRID["wd"],):
		run_id += 1
		print(f"\n=== Run {run_id}/{total_runs} ===")
		print(f"history={h}, lr={lr}, wd={wd}")

		best_tr, best_vl = train_model(h, lr, wd)

		print(f'Best results: training {best_tr} | validaiton {best_vl}')

		entry = {
			"history_len": h,
			"lr": lr,
			"weight_decay": wd,
			"best_train_qmse": best_tr,
			"best_val_qmse": best_vl,
			"overfit_gap": best_vl - best_tr,
			"timestamp": datetime.now().isoformat(),
		}

		results.append(entry)
		with open(CURRENT_ENV['data_dir'] + RESULTS_FILE, "w") as f:
			json.dump(results, f, indent=2)

	print("\nGrid search complete.")


			
if __name__ == "__main__":
	run_grid_search()

##### SOME PRELIMINARY RESULTS ####
# 512 4 4 128 -> 14.3% | 0.75 | 0.59