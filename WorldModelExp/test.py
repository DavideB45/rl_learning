import numpy as np
import torch
from PIL import Image

from helpers.general import best_device
from helpers.model_loader import load_vq_vae, load_lstm_quantized
from global_var import PUSHER
from envs.wrapper import PusherWrapEnv
import cv2

if __name__ == '__main__':
	vq = load_vq_vae(PUSHER, 64, 16, 4, True, True, best_device())
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), 1024, True, True, False)
	env = PusherWrapEnv(vq, lstm)
	env.reset()
	from helpers.data import make_seq_dataloader_safe, get_data_path
	from helpers.general import best_device
	tr_seq = make_seq_dataloader_safe(get_data_path(PUSHER['data_dir'], True, 0), vq, 100, 1)

	for i in tr_seq:
		print('doing', i['latent'].shape)
		for j in range(i['latent'].shape[1]):
			with torch.no_grad():
				print(j)
				img = vq.decode(i['latent'][:, j, :].to(vq.device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
				img = (img * 255).astype(np.uint8)
				image = Image.fromarray(img)
				image_resized = image.resize((512, 512), Image.NEAREST)
				cv2.imshow('Env', np.array(image_resized))
				cv2.waitKey(100)
			action = i['action'][0, j, :]
			observation, reward, terminated, truncated, info = env.step(action)
			env.render()
		exit()

	from dynamics.blocks_tr import Transformer, PositionalEncoding
	import torch as t

	EMB = 2
	LEN = 3
	BAT = 2
	enc = PositionalEncoding(emb_size=EMB, dropout=0, max_len=14)

	print('Seq Bat Emb')
	standard = enc(t.zeros([LEN, BAT, EMB]))
	for i in range(LEN):
		print(standard[i, :])

	enc = PositionalEncoding(emb_size=EMB, dropout=0, max_len=14, batch_first=True)
	print('Bat Seq Emb')
	bf = enc(t.zeros([BAT, LEN, EMB]))
	for i in range(LEN):
		print(bf[:, i, :])

	from dynamics.transformer import TransformerArc
	from helpers.general import best_device

	model = TransformerArc(
		in_size=3,
		emb_size=10,
		max_seq_len=20,
		n_heads=2,
		n_transformer=3,
		dropout=0.1,
		device=best_device()
	)

	print(model)