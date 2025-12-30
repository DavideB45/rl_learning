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
	vq = load_vq_vae(CURRENT_ENV, 256, 4, 4, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 1024)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 40, 0.2, 64)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE)
	best_q_mse = 10000
	begin = time()
	max_tr_err = max_error(tr)
	max_vl_err = max_error(vl)
	no_imporvemets = 0
	for i in range(200):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=5, err_decay=0.99)
		err_vl = lstm.eval_rwm_style(vl, init_len=5, err_decay=0.99)
		errors_str = f'{i}: mse:{err_tr['mse']:.4f} qmse:{err_tr['qmse']:.4f} || mse:{err_vl['mse']:.4f} qmse:{err_vl['qmse']:.4f}'
		if err_vl['qmse'] < best_q_mse:
			print('\033[94m' + errors_str + '\033[0m')
			perc_err = f'tr: {_perc(err_tr["qmse"], max_tr_err):.1f}% || vl: {_perc(err_vl["qmse"], max_vl_err):.1f}%'
			print('\033[95m' + perc_err + '\033[0m')
			save_lstm_quantized(CURRENT_ENV, lstm)
			best_q_mse = err_vl['qmse']
			no_imporvemets = 0
		else:
			no_imporvemets += 1
			print(errors_str, end='\n')
			if no_imporvemets >= 8:
				print('Early stopping for no improvements')
	end = time()
	print(f'Time elapsed {end - begin}')


##### SOME PRELIMINARY RESULTS ####
# 512 4 4 128 -> 14.3% | 0.75 | 0.59

# 1024 4 8 256:
# 
# training is very slow, but it can make sense to have an early stop after 5/7 epochs of no improvement
# although the training keeps imporving also after 150 epocs and can be valuable
# lr was set to be 2e-5 which is quite low indeed, but in previous experiments high lr was a problem
# maybe when we will use more data batch size can be increased and so can lr
# 
# trained with 40 steps, initialised with 5-6 error decay 0.99
# usual tr val split with 20% validation
#
# errors:
#	tr	|	vl	|
#	.33	|	.30	|
# 10.4% |  8.2% |
#
# 6867 sec () 1:54:27
# 4 seconds of sleep each epoch

# 1024 4 4 256:
# 
# Early stopping was not really stopping, same parameter as above
# Consideration: the paper that uses transformer world model uses a
# vector dimention for quantization of 512 obtaining a representation that is
# 4x4x512 which is something...
# 
# trained with 40 steps, initialised with 5-6 error decay 0.99
# usual tr val split with 20% validation
#
# errors:
#	tr	|	vl	|
#	.28	|	.30	|
#  7.9% |  8.4% |
#
# ---- sec () -:--:--
# mi sono disconnesso dal hbp-vulcano prima del termine
# i miglioramenti sembravano fermi dopo 140 epoche

# Note: the input (also when generated) is detached this can 
# stop additional gradient to flow through the LSTM