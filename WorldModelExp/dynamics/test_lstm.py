import torch
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

SEQ_LEN = 6
INIT_LEN = 2

if __name__ == '__main__':
	dev = best_device()
	vq = load_vq_vae(CURRENT_ENV, 64, 16, 4, True, dev)
	lstm = load_lstm_quantized(CURRENT_ENV, vq, dev, 1024, False, True, False)
	tr, vl = make_sequence_dataloaders(CURRENT_ENV['data_dir'], vq, 100, 0.5, 32, 200000)
	print(f'Number of parameter in LSTM: {lstm.param_count()}')
	print(f'Number of parameter in VQAE: {vq.param_count()}')
	print(f'Total number of parameter = {lstm.param_count() + vq.param_count()}')
	print(lstm.eval_epoch(vl))

	best_q_mse = 10000
	begin = time()

	with no_grad():
		lstm.eval()
		sequence = next(iter(tr))
		latent = sequence['latent'].to(dev)
		action = sequence['action'].to(dev)
		print(f'generating sequence given: {latent[:, 0:INIT_LEN, :, :, :].shape}')
		_, _, h = lstm.forward(latent[:, 0:INIT_LEN, :, :, :], action[:, 0:INIT_LEN :])
		_, generated, _ = lstm.ar_forward(latent[:, INIT_LEN:INIT_LEN+1, :, :, :], action[:, INIT_LEN:, :], h)


	end = time()
	print(f'Time elapsed {end - begin}')
	for b in range(2):
		latent_gt = latent[b, INIT_LEN + 1:, :, :, :]
		latent_pred = generated[b]

		print(f'latent shape: {latent_gt.shape}')
		print(f'generated sequence: {latent_pred.shape}')
		#plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10)
		save_video(vq, latent_gt, latent_pred, f"images/world_model_rollout_{b}.mp4")

	begin = time()
	with no_grad():
		lstm.eval()
		latent = sequence['latent'].to(dev)
		action = sequence['action'].to(dev)
		generated2 = []
		initializer = latent[:, 0:INIT_LEN, :, :, :]
		for_forward = latent[:, INIT_LEN:INIT_LEN+1, :, :, :]
		for i in range(action.shape[1] -2):
			_, _, h = lstm.forward(initializer, action[:, i:INIT_LEN+i, :])
			initializer = torch.cat([initializer, for_forward], dim=1)[:, 1:, :, :, :]
			_, for_forward, _ = lstm.ar_forward(for_forward, action[:, INIT_LEN+i:INIT_LEN+i+1, :], h)
			generated2.append(for_forward)
	generated2 = torch.cat(generated2, dim=1)
	end = time()
	print(f'Time elapsed {end - begin}')
	for b in range(2):
		latent_gt = latent[b, INIT_LEN + 1:, :, :, :]
		latent_pred = generated[b]

		print(f'latent shape: {latent_gt.shape}')
		print(f'generated sequence: {latent_pred.shape}')
		save_video(vq, latent_gt, latent_pred, f"images/world_model_rollout_copy_{b}.mp4")

