import torch
from itertools import product
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from helpers.general import best_device
from helpers.data import make_img_dataloader
from helpers.model_loader import save_vq_vae
from global_var import CURRENT_ENV

NUM_EPOCHS = 50
LEARNING_RATE = [1e-4, 4e-4]
WEIGTH_DECAY = [0, 0.001, 0.005]

LATENT_DIM_VQ = [4]
CODE_DEPTH = [16, 32]
CODEBOOK_SIZE = [64, 128]
EMA_MODE = [True]
SMOOTHING = [True, False]

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()



def trainOne(cs, cd, ld, ema, lr, wd, best_error, sm):
	vae = VQVAE(codebook_size=cs,
			code_depth=cd,
			latent_dim=ld,
			commitment_cost=0.25,
			device=DEVICE,
			ema_mode=ema
		)
	train_loader, val_loader = make_img_dataloader(data_dir=DATA_PATH, batch_size=128, test_split=0.2)
	best_val_loss = float('inf')
	optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)
	print(f'Parameters: cs {cs}, cd {cd}, ld {ld}, ema {ema}, lr {lr}, wd {wd} sm {sm}')
	
	# Create loss log file
	log_filename = f'{CURRENT_ENV["data_dir"]}histories/vq/losses_cs{cs}_cd{cd}_ld{ld}_ema{ema}_lr{lr}_wd{wd}_sm{sm}.csv'
	with open(log_filename, 'w') as f:
		f.write('Epoch,Train_total_loss,Val_total_loss,Train_commit_loss,Val_commit_loss,Train_codes_usage,Val_codes_usage,Train_recon_loss,Val_recon_loss,Train_flatness_loss,Val_flatness_loss\n')
	
	no_improvement_epochs = 0
	for epoch in range(NUM_EPOCHS):
		print("-" * 25 + f" {(epoch + 1):02}/{NUM_EPOCHS} " + "-" * 25)
		vae.train()
		tr_loss = vae.train_epoch(train_loader, optim, reg=1 if sm else 0)
		vae.eval()
		val_loss = vae.eval_epoch(val_loader, reg=1 if sm else 0)
		colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
		reset = '\033[0m'
		if val_loss['total_loss'] < best_val_loss:
			no_improvement_epochs = 0
			best_val_loss = val_loss['total_loss']
			if best_val_loss < best_error:
				print(f"{colors[-1]}  New best model saved!{reset}")
				save_vq_vae(CURRENT_ENV, vae)
				best_error = best_val_loss
		else:
			no_improvement_epochs += 1
			if no_improvement_epochs >= 10:
				print(f"{colors[0]}  No improvement for 10 epochs, stopping training.{reset}")
				break
			else:
				print(f"{colors[0]}  No improvement for {no_improvement_epochs} epochs.{reset}")
		
		print(f"  Train total_loss: {tr_loss['total_loss']:.4f}, Val total_loss: {val_loss['total_loss']:.4f}")
		print(f"  Train recon_loss: {tr_loss['recon_loss']:.4f}, Val recon_loss: {val_loss['recon_loss']:.4f}")
		print(f"  Train flatness_loss: {tr_loss['flatness_loss']:.4f}, Val flatness_loss: {val_loss['flatness_loss']:.4f}")
		with open(log_filename, 'a') as f:
			f.write(f"{epoch+1},{tr_loss['total_loss']},{val_loss['total_loss']},{tr_loss['commit_loss']},{val_loss['commit_loss']},{tr_loss['codes_usage']},{val_loss['codes_usage']},{tr_loss['recon_loss']},{val_loss['recon_loss']},{tr_loss['flatness_loss']},{val_loss['flatness_loss']}\n")
	return best_error

if __name__ == "__main__":
	
	time_start = time.time()
	for cs, cd, ld, sm in product(CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM_VQ, SMOOTHING):
				best_error = float('inf')
				for ema, lr, wd in product(EMA_MODE, LEARNING_RATE, WEIGTH_DECAY):
					log_filename = f'{CURRENT_ENV["data_dir"]}histories/vq/losses_cs{cs}_cd{cd}_ld{ld}_ema{ema}_lr{lr}_wd{wd}_sm{sm}.csv'
					
					# Check if model has already been trained
					if os.path.exists(log_filename):
						try:
							with open(log_filename, 'r') as f:
								lines = f.readlines()
								if len(lines) > 1:  # Has header + at least one epoch
									best_val_loss = float('inf')
									for line in lines[1:]:  # Skip header
										val_loss = float(line.strip().split(',')[2])
										if val_loss < best_val_loss:
											best_val_loss = val_loss
									print(f'Found existing log file: {log_filename}')
									print(f'Model already trained. Best val_loss: {best_val_loss:.4f}')
									if best_val_loss < best_error:
										best_error = best_val_loss
									continue
						except (IndexError, ValueError):
							pass  # If file is corrupted, retrain
					
					best_error = trainOne(cs, cd, ld, ema, lr, wd, best_error, sm)
	elapsed_time = time.time() - time_start
	hours, remainder = divmod(elapsed_time, 3600)
	minutes, seconds = divmod(remainder, 60)
	print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")