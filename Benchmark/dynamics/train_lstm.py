import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from helpers.data import make_seq_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV, LATENT_DIM, CODE_DEPTH, CODEBOOK_SIZE, SMOOTH
from global_var import HIDDEN_DIM, SEQ_LEN, INIT_LEN, LSTM_EPOCS, LSTM_LR, LSTM_WD

from torch.optim import Adam
from time import time


SMOOTHING = True if SMOOTH > 0 else False
if __name__ == '__main__':

	#torch.manual_seed(76)
	torch.manual_seed(7)
	torch.manual_seed(9)

	print(
		f"LR={LSTM_LR}, LREG={LSTM_WD}\n"
		f"CB={CODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 4, HIDDEN_DIM)
	tr = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, 0), vq, SEQ_LEN, 128)
	vl = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], False, 0), vq, SEQ_LEN, 128)
		
	optim = Adam(lstm.parameters(), lr=LSTM_LR, weight_decay=LSTM_WD)
	best_q_mse = 10000
	begin = time()
	no_imporvemets = 0
	print(f"Number of parameters: {lstm.param_count()/1e6:.2f}M")
	print(lstm.param_count())
	exit()
	for i in range(LSTM_EPOCS):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99)
		errors_str = f"{i}: mse:{err_tr['mse']:.4f} qmse:{err_tr['qmse']:.4f} || mse:{err_vl['mse']:.4f} qmse:{err_vl['qmse']:.4f}"
		if err_vl['mse'] < best_q_mse:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr: {err_tr["acc"]:.1f}% || vl: {err_vl["acc"]:.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm, tf=SMOOTHING)
			best_q_mse = err_vl['mse']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			print(errors_str, end='\n')
			if no_imporvemets >= 10:
				print('Early stopping for no improvements')
				break
	end = time()
	print(f'Time elapsed {end - begin}')
