from torch.optim import Adam
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import DataLoader
import time

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import make_image_dataloader_safe, make_seq_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae, load_lstm_quantized, save_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import PUSHER
from vae.vqVae import VQVAE
from dynamics.lstmc import LSTMQClass

from envs.simulator import PusherDreamEnv
from envs.wrapper import PusherWrapEnv, generate_data

# VAE RELATED PARAMETERS
SMOOTH 		= True
LATENT_DIM	= 4
CODE_DEPTH	= 32
CODEBOOK_S	= 64
USE_EMA		= True

# LSTM RELATED PARAMETERS
USE_KL 		= False
HIDDEN_DIM	= 512
SEQ_LEN		= 23
INIT_LEN	= 18
# (Smooth is not present becasuse needs to be consistent with the vq)

# PPO RELATED PARAMETERS
N_ROUNDS	= 5 # number of training iterations to do (the first data gathering counts)

colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
reset = '\033[0m'

def main():

	start_time = time.time()
	vq = VQVAE(CODEBOOK_S, CODE_DEPTH, LATENT_DIM, 0.25, best_device(), True)
	lstm = LSTMQClass(vq, best_device(), PUSHER['a_size'], 17, HIDDEN_DIM)
	agent = None
	with open('res.csv', 'w') as f:
			f.write(f'mean,var')

	for round in range(N_ROUNDS):
		print(f'Training round: {round}')
		generate_data(vq, lstm, 200, policy=agent, training_set=True)
		tr_seq = make_seq_dataloader_safe(get_data_path(PUSHER['data_dir'], True, 0), vq, SEQ_LEN, 128)
		generate_data(vq, lstm, 200, policy=agent, training_set=False)
		vl_seq = make_seq_dataloader_safe(get_data_path(PUSHER['data_dir'], False, 0), vq, SEQ_LEN, 128)

		vq = tune_vq(model=vq, num_epocs=1, reg=2 if SMOOTH else 0)
		lstm = tune_lstm(lstm, tr=tr_seq, vl=vl_seq, encoder=vq, num_epocs=1)
		wrapper_env = PusherWrapEnv(vq, lstm)
		dream_env = PusherDreamEnv(vq, lstm, vl_seq, 1, 20, 2)

		agent = tune_agent(agent, num_steps=20000, env=dream_env)
		grades = evaluate_policy(agent, wrapper_env, warn=False, n_eval_episodes=15)
		print(grades)

		with open('res.csv', 'a') as f:
			f.write(f'{grades[0]},{grades[1]}')
		print(f"\033[1;31m--- {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))} ---\033[0m")




def tune_vq(model:VQVAE, num_epocs:int=20, lr:float=1e-3, wd:float=1e-3, reg:float=2) -> VQVAE:
	tr = make_image_dataloader_safe(PUSHER['data_dir'], traininig=True)
	vl = make_image_dataloader_safe(PUSHER['data_dir'], traininig=False)
	optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
	best_val_loss = float('inf')
	no_improvements = 0
	for epoch in range(num_epocs):
		print("-" * 25 + f" {(epoch + 1):02}/{num_epocs} " + "-" * 25)
		tr_loss = model.train_epoch(tr, optim, reg)
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
	# this last line is needed: if the loop terminated with early stopping we still use the best model found
	del model
	return load_vq_vae(PUSHER, CODEBOOK_S, CODE_DEPTH, LATENT_DIM, USE_EMA, SMOOTH, best_device())

def tune_lstm(model: LSTMQClass, tr:DataLoader, vl:DataLoader, encoder: VQVAE, num_epocs:int=20, lr:float=5e-5, wd=5e-4) -> LSTMQClass:
	optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
	best_val_loss = float('inf')
	no_improvements = 0
	for epoch in range(num_epocs):
		err_tr = model.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		err_vl = model.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99, useKL=USE_KL)
		if err_vl['ce'] < best_val_loss:
			print_lstm_analytics(epoch, err_tr, err_vl)
			best_val_loss = err_vl['ce']
			no_improvements = 0
			save_lstm_quantized(PUSHER, model, cl=True, kl=USE_KL, tf=SMOOTH)
		else:
			no_improvements += 1
			if no_improvements >= 5:
				break
	del model
	return load_lstm_quantized(PUSHER, encoder, best_device(), HIDDEN_DIM, SMOOTH, True, USE_KL)
	
def tune_agent(agent:PPO, env:PusherDreamEnv, num_steps:int=100000) -> PPO:
	if agent is None:
		agent = PPO(MlpPolicy, env)
	agent = agent.learn(num_steps, progress_bar=True, reset_num_timesteps=False)
	agent.save(PUSHER['models'] + 'agent')
	return PPO.load(PUSHER['models'] + 'agent', env)




PURPLE = "\033[95m"; YELLOW = "\033[93m"; BLUE   = "\033[94m"; RESET  = "\033[0m"
COL1, COL2, COL3 = 15, 12, 12
WIDTH = COL1 + COL2 + COL3 + 6
def row(c1, c2="", c3="", color=RESET):
    print(color + f"| {c1:<{COL1}} | {c2:>{COL2}} | {c3:>{COL3}} |" + RESET)
def sep(color=RESET):
    print(color + "+" + "-"*(WIDTH+2) + "+" + RESET)
def print_lstm_analytics(epoch, err_tr, err_vl):
	sep(PURPLE)
	row(f"Epoch {epoch}", "Train", "Val", YELLOW)
	sep(PURPLE)
	row("CE",        f"{err_tr['ce']:.4f}",        f"{err_vl['ce']:.4f}",        BLUE)
	row("Prop MSE",  f"{err_tr['prop_mse']:.4f}",  f"{err_vl['prop_mse']:.4f}",  BLUE)
	row("Rew MSE", f"{err_tr['reward_mse']:.4f}",  f"{err_vl['reward_mse']:.4f}",  BLUE)
	row("Accuracy",  f"{err_tr['acc']*100:.1f}%",  f"{err_vl['acc']*100:.1f}%",  PURPLE)
	row("MSE",       f"{err_tr['mse']:.4f}",       f"{err_vl['mse']:.4f}",       PURPLE)
	row("First Acc", f"{err_tr['first_acc']*100:.1f}%",
					f"{err_vl['first_acc']*100:.1f}%", PURPLE)
	sep(PURPLE)

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