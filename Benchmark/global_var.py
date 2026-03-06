IMG_DIR = "imgs/"
TRANSITIONS = "action_reward_data.json"
MODELS_DIR = "models/"


BUTTON_DATA_DIR = "data/button-press/"
BUTTON = {
	"env_name": "button-press-v3",
	"img_dir": BUTTON_DATA_DIR + IMG_DIR,
	"models": BUTTON_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

CURRENT_ENV = BUTTON

LATENT_DIM = 8
CODE_DEPTH = 16
CODEBOOK_SIZE = 128
SMOOTH = 5
VQ_EPOCS = 4
VQ_LR = 1e-3
VQ_WD = 0.001

HIDDEN_DIM = 1024
SEQ_LEN = 25
INIT_LEN = 15
LSTM_EPOCS = 2
LSTM_LR = 5e-5
LSTM_WD = 1e-3

N_ROUNDS = 20
PPO_STEPS = 1000000