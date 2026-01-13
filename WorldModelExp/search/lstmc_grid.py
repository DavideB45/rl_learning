import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstmc import LSTMQClass
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from time import time

NUM_EPOCS=65 # this is (if there is no early stopping around 1 our per model)
LEARNING_RATE=1e-5
LAMBDA_REG = 2e-3
USE_KL = True

CDODEBOOK_SIZE = 256
CODE_DEPTH = 8
LATENT_DIM = 4

HIDDEN_DIM = 1024
SEQ_LEN = 7
INIT_LEN = 2

dev = best_device()

def make_lstm(lr:float, wd:float, kl:bool, hd:int, tr, vl) -> dict:
	lstm = LSTMQClass(vq, dev, CURRENT_ENV['a_size'], hd)
	optim = Adam(lstm.parameters(), lr=lr, weight_decay=wd)
	best_ce = 10000
	
	no_imporvemets = 0
	for i in range(200):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, useKL=kl)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=kl)
		errors_str = f'{i}: ce:{err_tr['ce']:.4f} mse:{err_tr['mse']:.4f} || ce:{err_vl['ce']:.4f} mse:{err_vl['mse']:.4f}'
		if err_vl['ce'] < best_ce:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr acc: {(err_tr["acc"]*100):.1f}% || vl acc: {(err_vl["acc"]*100):.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm, cl=True, kl=kl)
			best_ce = err_vl['ce']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			if no_imporvemets >= 5:
				break


if __name__ == '__main__':
	
	print(
		f"LR={LEARNING_RATE}, LREG={LAMBDA_REG}, KL={USE_KL}\n"
		f"CB={CDODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)

	
	# for vae in vae needs to be a tuple so it's easier to optimize (we will test only 2 probably)
	vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, dev)
	# for wd in [0, ...]
	# for kl in [true, false]
	# for lr in [1e-5, 2e-5]
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 1000000000)
	make_lstm(lr=LEARNING_RATE, wd=LAMBDA_REG, kl=USE_KL, hd=HIDDEN_DIM, tr=tr, vl=vl)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_REG)
	best_ce = 10000
	begin = time()
	no_imporvemets = 0
	err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
	print(err_vl)
	for i in range(200):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		errors_str = f'{i}: ce:{err_tr['ce']:.4f} mse:{err_tr['mse']:.4f} || ce:{err_vl['ce']:.4f} mse:{err_vl['mse']:.4f}'
		if err_vl['ce'] < best_ce:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr acc: {(err_tr["acc"]*100):.1f}% || vl acc: {(err_vl["acc"]*100):.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm, cl=True, kl=USE_KL)
			best_ce = err_vl['ce']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			print(errors_str, end='\n')
			if no_imporvemets >= 5:
				print('Early stopping for no improvements')
				break
	end = time()
	print(f'Time elapsed {end - begin}')