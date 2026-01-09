import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE
from helpers.general import best_device
from helpers.data import make_img_dataloader
from global_var import CURRENT_ENV

LATENT_DIM_VQ = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 128
EMA_MODE = True

NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

DATA_PATH = CURRENT_ENV['img_dir']
DEVICE = best_device()
train_loader, val_loader = make_img_dataloader(data_dir=DATA_PATH, batch_size=64, test_split=0.2)

def do_a_model(latentdim:int, depth:int, codebook:int, ema:bool, epochs:int, lr:float, wd:float) -> None:
	vae = VQVAE(codebook, depth, latentdim, 0.25, DEVICE, ema)
	op = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)
	history = {
		'tr':[],
		'vl':[]
	}
	for epoch in range(epochs):
		tr_loss = vae.train_epoch(train_loader, op)
		val_loss = vae.eval_epoch(val_loader)
		history['tr'].append(tr_loss)
		history['vl'].append(val_loss)

if __name__ == "__main__":
	vae = VQVAE(
		codebook_size=CODEBOOK_SIZE,
		code_depth=CODE_DEPTH,
		latent_dim=LATENT_DIM_VQ,
		commitment_cost=0.25,
		device=DEVICE,
		ema_mode=EMA_MODE
	)
	do_a_model(LATENT_DIM_VQ, CODE_DEPTH, CODEBOOK_SIZE, EMA_MODE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
	