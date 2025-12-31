import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstmc import LSTMQClass
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import no_grad
from time import time

LEARNING_RATE=2e-5

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 256, 8, 4, True, dev)
	lstm = LSTMQClass(vq, dev, CURRENT_ENV['a_size'], 1024)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 40, 0.2, 64, 1000000000)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE)
	best_ce = 10000
	begin = time()
	no_imporvemets = 0
	for i in range(200):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=5, err_decay=0.99)
		err_vl = lstm.eval_rwm_style(vl, init_len=5, err_decay=0.99)
		errors_str = f'{i}: ce:{err_tr['ce']:.4f} mse:{err_tr['mse']:.4f} || ce:{err_vl['ce']:.4f} mse:{err_vl['mse']:.4f}'
		if err_vl['ce'] < best_ce:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr acc: {(err_tr["acc"]*100):.1f}% || vl acc: {(err_vl["acc"]*100):.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm, cl=True)
			best_ce = err_vl['ce']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			print(errors_str, end='\n')
			if no_imporvemets >= 8:
				print('Early stopping for no improvements')
	end = time()
	print(f'Time elapsed {end - begin}')