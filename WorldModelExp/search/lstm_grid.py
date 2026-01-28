import json
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

VAE_TO_TEST = [(4, 16, 128), (4, 16, 64)] # latent, code_depth, codebook_size
NUM_EPOCS=200 # this is (if there is no early stopping around 1 our per model)
LEARNING_RATES=[1e-6, 5e-6, 1e-5, 2e-5]
LAMBDA_REGS = [0, 1e-4, 5e-4, 1e-3]

HIDDEN_DIM = 2048
SEQ_LEN = 23
INIT_LEN = 18

TOT = len(VAE_TO_TEST)*len(LEARNING_RATES)*len(LAMBDA_REGS)
dev = best_device()

def make_lstm(lr:float, wd:float, hd:int, tr, vl, min_err) -> tuple[dict, float]:
	'''
	Docstring for make_lstm
	
	:param lr: Learning rate
	:type lr: float
	:param wd: Weight decay for the Adam optimizer
	:type wd: float
	:param hd: Hidden dimension size for the LSTM
	:type hd: int
	:param tr: Training dataloader
	:type tr: torch.utils.data.DataLoader
	:param vl: Validation dataloader
	:type vl: torch.utils.data.DataLoader
	:param min_err: Minimum error to consider the model worth saving
	:return: History dictionary and minimum error achieved
	:rtype: tuple[dict, float]
	'''
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], hd)
	optim = Adam(lstm.parameters(), lr=lr, weight_decay=wd)

	history = {
		'tr':{
			'qmse':[],
			'mse':[],
			'acc':[]
		},
		'vl':{
			'qmse':[],
			'mse':[],
			'acc':[]
		},
	}
	
	curr_best = float('inf')
	no_imporvemets = 0
	for _ in range(NUM_EPOCS):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99)

		history['tr']['qmse'].append(err_tr['qmse'])
		history['tr']['mse'].append(err_tr['mse'])
		history['tr']['acc'].append(err_tr['acc'])
		history['vl']['qmse'].append(err_vl['qmse'])
		history['vl']['mse'].append(err_vl['mse'])
		history['vl']['acc'].append(err_vl['acc'])

		if err_vl['qmse'] < curr_best:
			curr_best = err_vl['qmse']
			no_imporvemets = 0
			if curr_best < min_err:
				save_lstm_quantized(CURRENT_ENV, lstm, cl=False)
				min_err = curr_best
		else:
			no_imporvemets += 1
			if no_imporvemets >= 10:
				break
	return history, min_err


if __name__ == '__main__':

	print(
		f"LR={LEARNING_RATES}, LREG={LAMBDA_REGS}\n"
		f"VAES={VAE_TO_TEST}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)
	
	# for vae in vae needs to be a tuple so it's easier to optimize (we will test only 2 probably)
	i = 0
	for ld, cd, cs in VAE_TO_TEST:
		vq = load_vq_vae(CURRENT_ENV, cs, cd, ld, True, dev)
		tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 1000000000000)
		min_err = float('inf')
		for lr in LEARNING_RATES:
			for wd in LAMBDA_REGS:
				start = time()
				history, min_err = make_lstm(lr=lr, wd=wd, hd=HIDDEN_DIM, tr=tr, vl=vl, min_err=min_err)
				# save the model history for future reference
				path = f"{CURRENT_ENV['data_dir']}histories/"
				if not os.path.exists(path):
					os.makedirs(path)
				version = f'lstm_{HIDDEN_DIM}_{ld}_{cd}_{cs}_{lr}_{wd}'
				with open(f"{path}{version}.json", "w") as f:
					json.dump(history, f, indent=4)
				end = time()
				i += 1
				print(f"Trained ({i}/{TOT}) LSTMC {version} in {(end - start)/60:.2f} minutes.")
				print(f"Current min found for {cs} = {min_err}")
