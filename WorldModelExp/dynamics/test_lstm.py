import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from dynamics.lstm import LSTMQuantized
from vae.vqVae import VQVAE
from helpers.data import make_sequence_dataloaders
from helpers.model_loader import load_vq_vae, load_lstm_quantized
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

def decode_latent_sequence(vq:VQVAE, latents:Tensor):
    """
    latents: (T, D, W, H)
    returns: (T, 3, H_img, W_img)
    """
    T = latents.size(0)
    with no_grad():
        imgs = vq.decode(latents)  # assumes batch decode
    return imgs

def plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10):
    """
    latent_gt:   (T+1, D, W, H)
    latent_pred: (T,   D, W, H)
    """

    T = min(max_frames, latent_pred.size(0))

    gt_imgs = decode_latent_sequence(vq, latent_gt[1:T+1])
    pred_imgs = decode_latent_sequence(vq, latent_pred[:T])

    gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().numpy()
    pred_imgs = pred_imgs.permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, T, figsize=(2*T, 4))

    for t in range(T):
        axes[0, t].imshow(gt_imgs[t])
        axes[0, t].set_title(f"GT t={t+1}")
        axes[0, t].axis("off")

        axes[1, t].imshow(pred_imgs[t])
        axes[1, t].set_title(f"Pred t={t+1}")
        axes[1, t].axis("off")

    plt.tight_layout()
    plt.show()

def save_video(vq, latent_gt, latent_pred, path="rollout.mp4", fps=5):
    """
    Saves a side-by-side GT vs prediction video
    """

    T = latent_pred.size(0)

    gt_imgs = decode_latent_sequence(vq, latent_gt[1:T+1])
    pred_imgs = decode_latent_sequence(vq, latent_pred)
    print(f'decoded sequence: {pred_imgs.shape}')

    gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().numpy()
    pred_imgs = pred_imgs.permute(0, 2, 3, 1).cpu().numpy()

    frames = []

    for t in range(T):
        gt = (gt_imgs[t] * 255).astype(np.uint8)
        pr = (pred_imgs[t] * 255).astype(np.uint8)
        frame = np.concatenate([gt, pr], axis=1)
        frames.append(frame)

    imageio.mimsave(path, frames)
    print(f"Saved video to {path}")

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

