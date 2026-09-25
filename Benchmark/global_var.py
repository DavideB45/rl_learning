IS_SERVER = False
EXP_ID = 2
LOG_NAME = f'res_{EXP_ID}'
GPU_ID = f"{EXP_ID%4}" # window 2 - peg 3

IMG_DIR = "imgs/"
TRANSITIONS = "action_reward_data.json"
MODELS_DIR = "models/"

if IS_SERVER:
	BASE = '/home/davide/github/rl_learning/Benchmark/'
else:
	BASE = '/Users/davide/Documents/github/rl_learning/Benchmark/'

REAL_SOFT_DATA_DIR = "data/real-soft/"
REAL_SOFT = {
	"env_name": "real-soft-v0",
	"img_dir": REAL_SOFT_DATA_DIR + IMG_DIR,
	"models": BASE + REAL_SOFT_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 3,
	"render_size": 64,
	"camera_id": 0,
}

CURRENT_ENV = REAL_SOFT

LATENT_DIM = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 32
SMOOTH = 5
VQ_EPOCS = 20 # used in learning loop for the initial training, then 1 epoch for each round
VQ_LR = 1e-3
VQ_WD = 0.001

EP_ON_LOOP = 20

SEQ_LEN = 25
INIT_LEN = 10
REW_WEIGHT = 1
USE_KL = True

HIDDEN_DIM = 1024
LSTM_EPOCS = 2 # used in learning loop for the initial training, then 1 epoch for each round
LSTM_LR = 5e-5
LSTM_WD = 1e-3
PROP_SIZE = 3


N_ROUNDS = 300 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/1000
PPO_STEPS = 100000
DREAM_LEN = 30
PPO_LR = 0.0003
ACTION_REPEAT = True
INIT_GATHER = 1010