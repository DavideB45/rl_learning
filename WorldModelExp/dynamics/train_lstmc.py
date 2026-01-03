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
	vq = load_vq_vae(CURRENT_ENV, 128, 4, 4, True, dev)
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

##########################
# 1024 4 8 256:
#
# 101: ce:0.0042 mse:0.0009 || ce:0.0139 mse:0.0299
# tr acc: 100.0% || vl acc: 99.8%
# tempo di allenamento mi pare uguale, lo ho fermato dopo circa un'oretta
# non mi sembra abbia senso andare oltre con il training
# questa è decisamente superiore alla versione che usa MSE

# 1024 4 4 128:
#
# Allenato usando la KL scritta forse in modo non eccellente (vedere se esistono implementazioni migliori)
# Il training è comparabile con quello del modello sopra tuttavia sono
# basati su VQVAE divresi quindi un confronto non può essere fatto
# 107: ce:0.7786 mse:1.1551 || ce:0.8166 mse:1.2540
# tr acc: 87.9% || vl acc: 87.4%
# stava ancora migliorando ma ero un po' stufo e volevo vedere come andava la ce

# 1024 4 4 128:
#
# Allenato usando la CE
#37: ce:0.4557 mse:1.6589 || ce:0.4789 mse:1.8746
# tr acc: 87.5% || vl acc: 86.6%
# 38: ce:0.4187 mse:1.4733 || ce:0.4388 mse:1.6487
# tr acc: 89.0% || vl acc: 88.2%
#
# 58: ce:0.0544 mse:0.1049 || ce:0.0721 mse:0.1980
# tr acc: 99.3% || vl acc: 98.8%
#
# 70: ce:0.0222 mse:0.0315 || ce:0.0351 mse:0.0938
# tr acc: 99.8% || vl acc: 99.4%
# l'ho stoppato perché dovevo andare a fare shopping e non penso possa milgiorare di molto...
