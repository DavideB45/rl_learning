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

LEARNING_RATE=2e-05
LAMBDA_REG = 0.00
USE_KL = False

CDODEBOOK_SIZE = 64
CODE_DEPTH = 16
LATENT_DIM = 4

HIDDEN_DIM = 1048
SEQ_LEN = 23
INIT_LEN = 18

PURPLE = "\033[95m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

COL1, COL2, COL3 = 15, 12, 12
WIDTH = COL1 + COL2 + COL3 + 6

def row(c1, c2="", c3="", color=RESET):
    print(
        color
        + f"| {c1:<{COL1}} | {c2:>{COL2}} | {c3:>{COL3}} |"
        + RESET
    )

def sep(color=RESET):
    print(color + "+" + "-"*(WIDTH+2) + "+" + RESET)

if __name__ == '__main__':
	
	print(
		f"LR={LEARNING_RATE}, LREG={LAMBDA_REG}, KL={USE_KL}\n"
		f"CB={CDODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)

	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, dev)
	lstm = LSTMQClass(vq, dev, CURRENT_ENV['a_size'], 1, HIDDEN_DIM)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 100)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_REG)
	best_ce = 10000
	begin = time()
	no_imporvemets = 0
	err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
	print(err_vl)
	for i in range(200):
		err_tr = lstm.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		err_vl = lstm.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		errors_str = f"{i}: ce:{err_tr['ce']:.4f} p mse:{err_tr['prop_mse']:.4f} || ce:{err_vl['ce']:.4f} p mse:{err_vl['prop_mse']:.4f}"
		if err_vl['ce'] < best_ce:
						
			sep(PURPLE)
			row(f"Epoch {i}", "Train", "Val", YELLOW)
			sep(PURPLE)

			row("CE",        f"{err_tr['ce']:.4f}",        f"{err_vl['ce']:.4f}",        BLUE)
			row("Prop MSE",  f"{err_tr['prop_mse']:.4f}",  f"{err_vl['prop_mse']:.4f}",  BLUE)
			row("Accuracy",  f"{err_tr['acc']*100:.1f}%",  f"{err_vl['acc']*100:.1f}%",  PURPLE)
			row("MSE",       f"{err_tr['mse']:.4f}",       f"{err_vl['mse']:.4f}",       PURPLE)
			row("First Acc", f"{err_tr['first_acc']*100:.1f}%",
							f"{err_vl['first_acc']*100:.1f}%", PURPLE)

			sep(PURPLE)
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