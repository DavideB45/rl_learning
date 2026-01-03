import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device
from helpers.dynamic_helper import plot_gt_vs_pred, save_video
from global_var import CURRENT_ENV

from torch import no_grad
from time import time

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 256, 8, 4, True, dev)
	lstm = load_lstm_quantized(CURRENT_ENV, vq, dev, 1024, False, True)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 99, 0.2, 2, 20)
	init_len = 5

	best_q_mse = 10000
	begin = time()

	with no_grad():
		lstm.eval()
		sequence = next(iter(vl))
		latent = sequence['latent'].to(dev)
		action = sequence['action'].to(dev)
		print(f'generating sequence given: {latent[:, 0:init_len, :, :, :].shape}')
		_, _, h = lstm.forward(latent[:, 0:init_len, :, :, :], action[:, 0:init_len :])
		_, generated, _ = lstm.ar_forward(latent[:, init_len:init_len+1, :, :, :], action[:, init_len:, :], h)


	end = time()
	print(f'Time elapsed {end - begin}')
	for b in range(2):
		latent_gt = latent[b, init_len + 1:, :, :, :]
		latent_pred = generated[b]

		print(f'latent shape: {latent_gt.shape}')
		print(f'generated sequence: {latent_pred.shape}')
		plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10)
		save_video(vq, latent_gt, latent_pred, f"world_model_rollout_{b}.mp4")

