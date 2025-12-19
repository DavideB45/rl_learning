import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import no_grad
from time import time, sleep

LEARNING_RATE=2e-5

def max_error(loader:DataLoader) -> float:
	total_dist = 0
	with no_grad():
		for batch in loader:
			latent = batch['latent']
			dist = F.mse_loss(latent[:, 1:, :, :, :], latent[:, :-1, :, :, :], reduction='mean')# / output.size(0)
			total_dist += dist.item()
	return total_dist/len(loader)

def _perc(amount:float, maxim:float) -> float:
	return (amount/maxim)*100

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 128, 4, 4, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 512)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 10, 0.2, 64)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE)
	best_q_mse = 10000
	begin = time()
	max_tr_err = max_error(tr)
	max_vl_err = max_error(vl)
	for i in range(200):
		sleep(4)
		err_tr = lstm.train_epoch(tr, optim, True)
		err_vl = lstm.eval_epoch(vl, True)
		errors_str = f'{i}: mse:{err_tr['mse']:.4f} qmse:{err_tr['qmse']:.4f} || mse:{err_vl['mse']:.4f} qmse:{err_vl['qmse']:.4f}'
		if err_vl['qmse'] < best_q_mse:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr: {_perc(err_tr["qmse"], max_tr_err):.1f}% || vl: {_perc(err_vl["qmse"], max_vl_err):.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm)
			best_q_mse = err_vl['qmse']
		else:
			print(errors_str, end='\n')
	end = time()
	print(f'Time elapsed {end - begin}')


##### SOME PRELIMINARY RESULTS ####
# 512 4 4 128 -> 14.3% | 0.75 | 0.59