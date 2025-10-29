import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import CURRENT_ENV
from vae import VAE
from dataset_func import make_sequence_dataloaders

make_sequence_dataloaders(data_file=CURRENT_ENV['transitions'])