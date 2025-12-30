import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.fnn import FNN
from vae.vqVae import VQVAE
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_fnn
from helpers.general import best_device
from helpers.dynamic_helper import plot_gt_vs_pred, save_video
from global_var import CURRENT_ENV

from torch import no_grad
from time import time

if __name__ == '__main__':
	history_len = 5
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 128, 4, 4, True, dev)
	dyn_fnn = load_fnn(CURRENT_ENV, vq, dev, history_len)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 99, 0.2, 2, 20)

	best_q_mse = 10000
	begin = time()

	with no_grad():
		dyn_fnn.eval()
		sequence = next(iter(tr))
		latent = sequence['latent'].to(dev)
		action = sequence['action'].to(dev)
		print(f'generating sequence given: {latent[:, 0:history_len, :, :, :].shape}')
		generated = dyn_fnn.ar_forward(latent[:, 0:history_len, :, :, :], action[:, (history_len-1):, :])

		print(f'Generated: {generated.shape}')
		
	end = time()
	print(f'Time elapsed {end - begin}')
	for b in range(2):
		latent_gt = latent[b]
		latent_pred = generated[b]

		print(f'latent shape: {latent_gt.shape}')
		print(f'generated sequence: {latent_pred.shape}')
		plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10)
		save_video(vq, latent_gt, latent_pred, f"world_model_rollout_{b}.mp4")