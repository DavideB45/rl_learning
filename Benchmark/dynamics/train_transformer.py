import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.transformer import TransformerArc
from helpers.data import make_seq_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae
from helpers.general import best_device
from global_var import *

from torch.optim import Adam
from time import time

if __name__ == '__main__':

	print(
		f"LR={TR_LR}, LREG={TR_WD}\n"
		f"CB={CODEBOOK_SIZE}, DEPTH={CODE_DEPTH}, LAT={LATENT_DIM}, HID={EMB_SIZE}\n"
		f"SEQ_LEN={SEQ_LEN}, INIT_LEN={INIT_LEN + 1}, LOSS ON {SEQ_LEN - INIT_LEN} steps"
	)
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, True if SMOOTH > 0 else False, dev)
	model = TransformerArc(CURRENT_ENV['a_size'], vq, EMB_SIZE, SEQ_LEN + 1, NUM_HEADS, NUM_LAYERS, 0.1, dev)
	tr = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, 0), vq, SEQ_LEN, max_ep=100000)
	vl = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], False, 0), vq, SEQ_LEN, max_ep=100000)
	

	optim = Adam(model.parameters(), lr=TR_LR, weight_decay=TR_WD)
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