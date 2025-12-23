import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.fnn import FNN
from vae.vqVae import VQVAE
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_fnn
from helpers.general import best_device
from global_var import CURRENT_ENV

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torch import no_grad, Tensor
from time import time
import matplotlib.pyplot as plt
import imageio
import numpy as np

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 128, 4, 4, True, dev)
	lstm = LSTMQuantized(vq, dev, CURRENT_ENV['a_size'], 512)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 99, 0.2, 2, 20)

	best_q_mse = 10000
	begin = time()

	with no_grad():
		lstm.eval()
		sequence = next(iter(tr))
		latent = sequence['latent'].to(dev)
		action = sequence['action'].to(dev)
		print(f'generating sequence given: {latent[:, 0:1, :, :, :].shape}')
		generated, _ = lstm.autoregressive_feed(latent[:, 0:1, :, :, :], action)
		_, fake_gen, _ = lstm.forward(latent[:, :-1, :, :, :], action)

		print(f'Generated: {generated.shape}')
		for i in range(4):
			print(f'Others    dist: {F.mse_loss(generated[:, i:(i + 1), :, :, :], generated[:, i+1:(i + 2), :, :, :], reduction="mean")}')
			print(f'Generated dist: {F.mse_loss(generated[:, i:(i + 1), :, :, :], latent[:, i:(i + 1), :, :, :], reduction="mean")}')
			print(f'Forwarded dist: {F.mse_loss(fake_gen[:, i:(i + 1), :, :, :], latent[:, i:(i + 1), :, :, :], reduction="mean")}')

	end = time()
	print(f'Time elapsed {end - begin}')
	for b in range(2):
		latent_gt = latent[b]
		latent_pred = generated[b]

		print(f'latent shape: {latent_gt.shape}')
		print(f'generated sequence: {latent_pred.shape}')
		plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10)
		save_video(vq, latent_gt, latent_pred, f"world_model_rollout_{b}.mp4")