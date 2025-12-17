import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from time import time

LEARNING_RATE=1e-5

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 256, 16, 8, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 512)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 10, 0.2, 64)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE)
	best_q_mse = 10000
	begin = time()
	for i in range(200):
		err_tr = lstm.train_epoch(tr, optim)
		err_vl = lstm.eval_epoch(vl)
		errors_str = f'{i}: mse:{err_tr['mse']:.4f} qmse:{err_tr['qmse']:.4f} || mse:{err_vl['mse']:.4f} qmse:{err_vl['qmse']:.4f}'
		if err_vl['qmse'] < best_q_mse:
			print('\033[94m' + errors_str + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm)
			best_q_mse = err_vl['qmse']
		else:
			print(errors_str, end='\n')
	end = time()
	print(f'Time elapsed {end - begin}')


##### SOME PRELIMINARY RESULTS ####
# 512 4 8 256 -> 5.50