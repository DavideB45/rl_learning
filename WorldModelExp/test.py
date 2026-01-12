from global_var import CURRENT_ENV

from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae
from helpers.general import best_device

LEARNING_RATE=1e-5
LAMBDA_REG = 2e-3
USE_KL = True

CDODEBOOK_SIZE = 128
CODE_DEPTH = 4
LATENT_DIM = 4

HIDDEN_DIM = 1024
SEQ_LEN = 8 # which would be 8 actions and 9 images
INIT_LEN = 2

dev = best_device()
vq = load_vq_vae(CURRENT_ENV, CDODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, dev)
tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, SEQ_LEN, 0.1, 1, 1000000000)

print(f'tr len: {len(tr)}')
print(f'vl len: {len(vl)}')