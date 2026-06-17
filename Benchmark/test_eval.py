import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from global_var import *
if 'MUJOCO_GL' not in os.environ:
	os.environ['MUJOCO_GL'] = 'egl'
	os.environ['MUJOCO_EGL_DEVICE_ID'] = GPU_ID
	os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

from helpers.data import make_seq_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from helpers.general import best_device

SMOOTHING = True if SMOOTH > 0 else False

encoder = load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, best_device())
model = load_lstm_quantized(CURRENT_ENV, encoder, best_device(), HIDDEN_DIM, SMOOTHING, cl=False, kl=False)
print(f"E.t = {encoder.training} M.t = {model.training} M.E.t = {model.quantizer.training}")
model.train()
print(f"E.t = {encoder.training} M.t = {model.training} M.E.t = {model.quantizer.training}")
model.eval()
print(f"E.t = {encoder.training} M.t = {model.training} M.E.t = {model.quantizer.training}")
model.train()
model.quantizer.eval()
print(f"E.t = {encoder.training} M.t = {model.training} M.E.t = {model.quantizer.training}")


print("TESTING DATA LOADER WEIGHTED")
tr_seq = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, EXP_ID), encoder, SEQ_LEN, 128, max_ep=5)