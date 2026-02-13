from torch.optim import Adam
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import make_sequence_dataloaders, make_image_dataloader_safe
from helpers.model_loader import load_vq_vae, load_lstm_quantized, save_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER
from vae.vqVae import VQVAE
from dynamics.lstmc import LSTMQClass

from envs.simulator import PusherDreamEnv
from envs.wrapper import PusherWrapEnv, generate_data

# VAE RELATED PARAMETERS
SMOOTH 		= False
LATENT_DIM	= 4
CODE_DEPTH	= 16
CODEBOOK_S	= 64
USE_EMA		= True

# LSTM RELATED PARAMETERS
USE_KL 		= False
HIDDEN_DIM	= 1024
# (Smooth is not present becasuse needs to be consistent with the vq)

colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
reset = '\033[0m'

def main():
	vq = load_vq_vae(PUSHER, CODEBOOK_S, CODE_DEPTH, LATENT_DIM, USE_EMA, SMOOTH, best_device()) # ricaricare ogni volta per tenere il meglio
	lstm = load_lstm_quantized(PUSHER, vq, best_device(), HIDDEN_DIM, SMOOTH, True, USE_KL) # ricaricare ogni volta per tenere il meglio
	dream_env = PusherDreamEnv(vq, lstm, 10, 200000) # da cambiare come vengono caricati i dati
	model = PPO(MlpPolicy, dream_env, verbose=0) # deve essere cambiato ogni volta?
	tune_vq()

def tune_vq(model:VQVAE, num_epocs:int=20, lr:float=1e-3, wd:float=1e-3, reg:float=2) -> VQVAE:
	tr = make_image_dataloader_safe(PUSHER['img_dir'], traininig=True)
	vl = make_image_dataloader_safe(PUSHER['img_dir'], traininig=False)
	optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
	best_val_loss = float('inf')
	no_improvements = 0
	for epoch in range(num_epocs):
		print("-" * 25 + f" {(epoch + 1):02}/{num_epocs} " + "-" * 25)
		model.train()
		tr_loss = model.train_epoch(tr, optim, reg)
		model.eval()
		val_loss = model.eval_epoch(vl, reg)
		if val_loss['total_loss'] < best_val_loss:
			best_val_loss = val_loss['total_loss']
			save_vq_vae(PUSHER, model, smooth=True if reg > 0 else False)
			print(f"{colors[-1]}  New best model saved!{reset}")
		else:
			no_improvements += 1
			if no_improvements >= 5:
				break
		for i, key in enumerate(tr_loss):
			color = colors[i % len(colors)]
			print(f"{color}  Train {key}: {tr_loss[key]:.4f}, Val {key}: {val_loss[key]:.4f}{reset}")
	

if __name__ == '__main__':
	main()
	


# STEPS
# 1 - gather some amount of data
# 2 - train a vector quantizer variational autoencoder
# 3 - train an lstmc
# 4 - a loop of some length begins
# 4.1 - train a PPO in the dream
# 4.2 - use the PPO to obtain data from a wrapped env
# 4.3 - tune the vq-vae
# 4.4 - tune the lstmc (more than the vq-vae)
# 5 fine