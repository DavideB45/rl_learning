import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam

LEARNING_RATE=1e-4

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 256, 8, 4, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 512)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 10, 0.2, 64)

	optim = Adam(lstm.parameters(), lr=LEARNING_RATE)
	for i in range(30):
		err = lstm.train_epoch(tr, optim)
		print(f'{i}: {err}')

