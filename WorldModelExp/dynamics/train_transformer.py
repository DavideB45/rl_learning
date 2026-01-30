import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.transformer import TransformerArc
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import no_grad
from time import time

LEARNING_RATE=1e-5
LAMBDA_REG = 0e-3

CDODEBOOK_SIZE = 64
CODE_DEPTH = 16
LATENT_DIM = 4

HIDDEN_DIM = 12
NUM_HEADS = 1
NUM_TRANSFORMERS = 1

SEQ_LEN = 7
INIT_LEN = 6

if __name__ == '__main__':

	print(
		f"LR={LEARNING_RATE}, LREG={LAMBDA_REG}\n"
		f"CB={CDODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={HIDDEN_DIM}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, dev)
	model = TransformerArc(CURRENT_ENV['a_size'], vq, HIDDEN_DIM, SEQ_LEN+1, NUM_HEADS, NUM_TRANSFORMERS, 0.1, dev)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 64, 300)

	optim = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_REG)
	best_mse = 10000
	begin = time()
	no_imporvemets = 0
	#print(model)
	#exit()
	for i in range(200):
		err_vl = model.train_epoch(tr, optim)
		print(i, ':', err_vl)
	end = time()
	print(f'Time elapsed {end - begin}')