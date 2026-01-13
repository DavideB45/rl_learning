import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstmc import LSTMQClass
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam


VAE_TO_TEST = [(4, 16, 128), (4, 16, 64)] # latent, code_depth, codebook_size
NUM_EPOCS=65 # this is (if there is no early stopping around 1 our per model)
LEARNING_RATES=[1e-5, 2e-5]
LAMBDA_REGS = [0, 1e-3, 2e-3]
USE_KL = [True, False]

HIDDEN_DIM = 1024
SEQ_LEN = 7
INIT_LEN = 2

dev = best_device()

def make_lstm(lr:float, wd:float, kl:bool, hd:int, tr, vl) -> dict:
	lstm = LSTMQClass(vq, dev, CURRENT_ENV['a_size'], hd)
	optim = Adam(lstm.parameters(), lr=lr, weight_decay=wd)
	best_ce = 10000

	history = {
		'tr':{
			'ce':[],
			'mse':[],
			'acc':[]
		},
		'vl':{
			'ce':[],
			'mse':[],
			'acc':[]
		},
	}
	
	no_imporvemets = 0
	for _ in range(NUM_EPOCS):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, useKL=kl)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=kl)

		history['tr']['ce'].append(err_tr['ce'])
		history['tr']['mse'].append(err_tr['mse'])
		history['tr']['acc'].append(err_tr['acc'])
		history['vl']['ce'].append(err_vl['ce'])
		history['vl']['mse'].append(err_vl['mse'])
		history['vl']['acc'].append(err_vl['acc'])
		
		if err_vl['ce'] < best_ce:
			best_ce = err_vl['ce']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			if no_imporvemets >= 5:
				break
	return history


if __name__ == '__main__':
	
	# for vae in vae needs to be a tuple so it's easier to optimize (we will test only 2 probably)
	for ld, cd, cs in VAE_TO_TEST:
		vq = load_vq_vae(CURRENT_ENV, cs, cd, ld, True, dev)
		tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 10)
		for lr in LEARNING_RATES:
			for wd in LAMBDA_REGS:
				for kl in USE_KL:
					history = make_lstm(lr=lr, wd=wd, kl=kl, hd=HIDDEN_DIM, tr=tr, vl=vl)

					# save the model history for future reference
					path = f"{CURRENT_ENV['data_dir']}histories/"
					if not os.path.exists(path):
						os.makedirs(path)
					version = f'lstmc_{HIDDEN_DIM}_{ld}_{cd}_{cs}_{kl}_{lr}_{wd}'
					with open(f"{path}{version}.json", "w") as f:
						json.dump(history, f, indent=4)
