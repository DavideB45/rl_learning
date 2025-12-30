import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from vae.vqVae import VQVAE

from torch import no_grad, Tensor
import matplotlib.pyplot as plt
import imageio
import numpy as np

def decode_latent_sequence(vq:VQVAE, latents:Tensor):
	'''
	decode a latent sequence using a VQVAE
	
	:param vq: the VQVAE model used to decode the latents (T, 3, H_img, W_img)
	:type vq: VQVAE
	:param latents: The latents to decode (T, D, W, H)
	:type latents: Tensor
	'''
	T = latents.size(0)
	with no_grad():
		imgs = vq.decode(latents)  # assumes batch decode
	return imgs

def plot_gt_vs_pred(vq, latent_gt, latent_pred, max_frames=10):
	'''
	Show a number of original against the prediction
	
	:param vq: The VQVAE model to decode the latents
	:param latent_gt: The original latents (T, 3, H_img, W_img)
	:param latent_pred: The predicted latents (T, 3, H_img, W_img)
	:param max_frames: The maximum amount of images to show
	'''
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
	'''
	Saves a side-by-side GT vs prediction video
	
	:param vq: The VQVAE model to decode the latents
	:param latent_gt: The original latents (T, 3, H_img, W_img)
	:param latent_pred: The predicted latents (T, 3, H_img, W_img)
	:param path: The path to which the video will be saved
	:param fps: fps
	'''

	T = latent_pred.size(0)

	gt_imgs = decode_latent_sequence(vq, latent_gt)
	pred_imgs = decode_latent_sequence(vq, latent_pred)
	print(f'decoded sequence: {pred_imgs.shape}')

	gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().numpy()
	pred_imgs = pred_imgs.permute(0, 2, 3, 1).cpu().numpy()

	frames = []

	for t in range(T):
		gt = (gt_imgs[t] * 255).astype(np.uint8)
		pr = (pred_imgs[t] * 255).astype(np.uint8)
		gt = gt.repeat(4, axis=0).repeat(4, axis=1)
		pr = pr.repeat(4, axis=0).repeat(4, axis=1)
		frame = np.concatenate([gt, pr], axis=1)
		frames.append(frame)

	imageio.mimsave(path, frames)
	print(f"Saved video to {path}")