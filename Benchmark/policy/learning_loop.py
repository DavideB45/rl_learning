from torch.optim import Adam
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo import PPO
from torch.utils.data import DataLoader
import time
import json

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from helpers.data import make_image_dataloader_safe, make_seq_dataloader_safe, get_data_path
from helpers.model_loader import load_vq_vae, load_lstm_quantized, save_vq_vae, save_lstm_quantized
from helpers.general import best_device
from global_var import *
from vae.vqVae import VQVAE
from dynamics.lstm import LSTMQuantized

from envs.simulator import MetaDreamEnv
from envs.wrapper import MetaWrapEnv, evaluate_gathering, generate_data

SMOOTHING = True if SMOOTH > 0 else False
# PPO RELATED PARAMETERS
policy_kwargs = dict(
	net_arch=dict(
		pi=[512, 512, 256],   # policy network layers
		vf=[521, 512, 256]    # value network layers
	),
	ortho_init=True
)

colors = ['\033[91m', '\033[95m', '\033[92m', '\033[93m', '\033[96m']
reset = '\033[0m'

def main():
	collecting_time = 0
	vq_training_time = 0
	lstm_training_time = 0
	dataset_generation_time = 0
	agent_training_time = 0

	start_time = time.time()
	vq = VQVAE(CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, 0.25, best_device(), True)
	lstm = LSTMQuantized(vq, best_device(), CURRENT_ENV['a_size'], 4, HIDDEN_DIM)
	agent = None
	with open('res.csv', 'w') as f:
			f.write(f'mrew,success\n')
	
	collecting_time -= time.time()
	generate_data(vq, lstm, n_sample=10000, training_set=True)
	generate_data(vq, lstm, n_sample=1000, training_set=False)
	collecting_time += time.time()

	for round in range(N_ROUNDS):
		print(f'Training round: {round + 1} of {N_ROUNDS}')

		vq_training_time -= time.time()
		vq = tune_vq(model=vq, num_epocs=VQ_EPOCS if round == 0 else 1, lr=VQ_LR, reg=SMOOTH, wd=VQ_WD)
		vq_training_time += time.time()

		dataset_generation_time -= time.time()
		tr_seq = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, 0), vq, SEQ_LEN, 128, max_ep=EP_ON_LOOP)
		vl_seq = make_seq_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], False, 0), vq, SEQ_LEN, 128, max_ep=15)
		dataset_generation_time += time.time()
		lstm_training_time -= time.time()
		lstm = tune_lstm(lstm, tr=tr_seq, vl=vl_seq, encoder=vq, num_epocs=LSTM_EPOCS if round == 0 else 1, lr=LSTM_LR, wd=LSTM_WD)
		lstm_training_time += time.time()
		
		dream_env = MetaDreamEnv(vq, lstm, vl_seq, init_len=INIT_LEN, ep_len=50, num_envs=50)
		agent_training_time -= time.time()
		agent = tune_agent(agent, num_steps=PPO_STEPS, env=dream_env)
		agent_training_time += time.time()

		collecting_time -= time.time()
		rew, succ = evaluate_gathering(vq, lstm, n_sample=1000, policy=agent, training_set=True)
		if round % 10 == 0:
			generate_data(vq, lstm, n_sample=1000, policy=agent, training_set=False)
		print(f"Average reward: {(sum(rew) / len(rew)):.2f}, Success rate: {(sum(succ) / len(succ)):.2%}")
		with open('res.csv', 'a') as f:
			for i in range(len(rew)):
				f.write(f'{rew[i]},{succ[i]}\n')
		collecting_time += time.time()

		print(f"\033[1;31m--- {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))} ---\033[0m")
	with open('time.json', 'w') as f:
		json.dump({
			'collecting_time': collecting_time,
			'vq_training_time': vq_training_time,
			'lstm_training_time': lstm_training_time,
			'dataset_generation_time': dataset_generation_time,
			'agent_training_time': agent_training_time,
		}, f, indent=1)



def tune_vq(model:VQVAE, num_epocs:int=20, lr:float=1e-3, wd:float=1e-3, reg:float=1) -> VQVAE:
	tr = make_image_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], True, 0), max_size=EP_ON_LOOP*500)
	vl = make_image_dataloader_safe(get_data_path(CURRENT_ENV['img_dir'], False, 0), max_size=1500)
	optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
	best_val_loss = float('inf')
	no_improvements = 0
	for epoch in range(num_epocs):
		print("-" * 25 + f" {(epoch + 1):02}/{num_epocs} " + "-" * 25)
		tr_loss = model.train_epoch(tr, optim, reg)
		val_loss = model.eval_epoch(vl, reg)
		if val_loss['total_loss'] < best_val_loss:
			best_val_loss = val_loss['total_loss']
			save_vq_vae(CURRENT_ENV, model, smooth=SMOOTHING)
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
	return load_vq_vae(CURRENT_ENV, CODEBOOK_SIZE, CODE_DEPTH, LATENT_DIM, True, SMOOTHING, best_device())

def tune_lstm(model: LSTMQuantized, tr:DataLoader, vl:DataLoader, encoder: VQVAE, num_epocs:int=20, lr:float=5e-5, wd=5e-4) -> LSTMQuantized:
	model.quantizer = encoder
	optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
	best_val_loss = float('inf')
	no_improvements = 0
	for epoch in range(num_epocs):
		err_tr = model.train_rwm_style(tr, optim, init_len=INIT_LEN, err_decay=0.99)#, useKL=USE_KL
		err_vl = model.eval_rwm_style(vl, init_len=INIT_LEN, err_decay=0.99)#, useKL=USE_KL
		if err_vl['mse'] < best_val_loss:
			print_lstm_analytics(epoch, err_tr, err_vl)
			best_val_loss = err_vl['mse']
			no_improvements = 0
			save_lstm_quantized(CURRENT_ENV, model, cl=False, kl=False, tf=SMOOTHING)
		else:
			no_improvements += 1
			if no_improvements >= 5:
				break
	del model
	return load_lstm_quantized(CURRENT_ENV, encoder, best_device(), HIDDEN_DIM, SMOOTHING, cl=False, kl=False)
	
def tune_agent(agent:PPO, env:MetaDreamEnv, num_steps:int=100000) -> PPO:
	if agent is None:
		agent = PPO(MlpPolicy, env, policy_kwargs=policy_kwargs, n_steps=500, batch_size=1000, learning_rate=0.0003, ent_coef=0.01, sde_sample_freq=10, use_sde=True)
	else:
		agent = PPO.load(CURRENT_ENV['models'] + 'agent', env)
	agent = agent.learn(num_steps, progress_bar=False, reset_num_timesteps=False)
	agent.save(CURRENT_ENV['models'] + 'agent')
	return agent




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
	row("MSE",		f"{err_tr['mse']:.4f}",			f"{err_vl['mse']:.4f}",			BLUE)
	row("QMSE",		f"{err_tr['qmse']:.4f}",		f"{err_vl['qmse']:.4f}",		BLUE)
	row("Prop MSE",	f"{err_tr['prop_mse']:.4f}",	f"{err_vl['prop_mse']:.4f}",	BLUE)
	row("Rew MSE",	f"{err_tr['reward_mse']:.4f}",	f"{err_vl['reward_mse']:.4f}",	BLUE)
	row("Accuracy",	f"{err_tr['acc']:.1f}%",		f"{err_vl['acc']:.1f}%",		PURPLE)
	row("First Acc",f"{err_tr['first_acc']*100:.1f}%",
					f"{err_vl['first_acc']*100:.1f}%", PURPLE)
	sep(PURPLE)

def print_lstmc_analytics(epoch, err_tr, err_vl):
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
