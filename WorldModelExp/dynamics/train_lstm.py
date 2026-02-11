import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from time import time

LEARNING_RATE=5e-5
LAMBDA_REG = 1e-3
NUM_EPOCS = 200

CDODEBOOK_SIZE = 64
CODE_DEPTH = 16
LATENT_DIM = 4
SMOOTHING = True

HIDDEN_DIM = 1024
SEQ_LEN = 23
INIT_LEN = 18

if __name__ == '__main__':

	#torch.manual_seed(76)
	torch.manual_seed(7)
	torch.manual_seed(9)

	print(
		f"LR={LEARNING_RATE}, LREG={LAMBDA_REG}\n"
		f"CB={CDODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], HIDDEN_DIM)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 300)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_REG)
	best_q_mse = 10000
	begin = time()
	no_imporvemets = 0
	for i in range(NUM_EPOCS):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99)
		errors_str = f"{i}: mse:{err_tr['mse']:.4f} qmse:{err_tr['qmse']:.4f} || mse:{err_vl['mse']:.4f} qmse:{err_vl['qmse']:.4f}"
		if err_vl['qmse'] < best_q_mse:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr: {err_tr["acc"]:.1f}% || vl: {err_vl["acc"]:.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm)
			best_q_mse = err_vl['qmse']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			print(errors_str, end='\n')
			if no_imporvemets >= 10:
				print('Early stopping for no improvements')
				break
	end = time()
	print(f'Time elapsed {end - begin}')
