EXP_ID = 0
LOG_NAME = f'res_{EXP_ID}'
GPU_ID = f"{EXP_ID%4}" # window 2 - peg 3

IMG_DIR = "imgs/"
TRANSITIONS = "action_reward_data.json"
MODELS_DIR = "models/"


BUTTON_DATA_DIR = "data/button-press/"
BUTTON = {
	"env_name": "button-press-v3",
	"img_dir": BUTTON_DATA_DIR + IMG_DIR,
	"models": BUTTON_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 64,
	"camera_id": 1,
}

PUSH_DATA_DIR = "data/push/"
PUSH = {
	"env_name": "push-v3",
	"img_dir": PUSH_DATA_DIR + IMG_DIR,
	"models": PUSH_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 96,
	"camera_id": 2,
}

DRAWERO_DATA_DIR = "data/drawer-open/"
DRAWER_OPEN = {
	"env_name": "drawer-open-v3",
	"img_dir": DRAWERO_DATA_DIR + IMG_DIR,
	"models": DRAWERO_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

PEG_DATA_DIT = "data/peg-insert/"
PEG_INSERT = {
	"env_name": "peg-insert-side-v3",
	"img_dir": PEG_DATA_DIT + IMG_DIR,
	"models": PEG_DATA_DIT + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

HAMMER_DATA_DIR = "data/hammer/"
HAMMER = {
	"env_name": "hammer-v3",
	"img_dir": HAMMER_DATA_DIR + IMG_DIR,
	"models": HAMMER_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 96,
	"camera_id": 2,
}

PICKB_DATA_DIR = "data/bin-pick/"
PICK_BIN = {
	"env_name": "bin-picking-v3",
	"img_dir": PICKB_DATA_DIR + IMG_DIR,
	"models": PICKB_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 96,
	"camera_id": 2,
}

PICKP_DATA_DIR = "data/pick-place/"
PICK_PLACE = {
	"env_name": "pick-place-v3",
	"img_dir": PICKP_DATA_DIR + IMG_DIR,
	"models": PICKP_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 96,
	"camera_id": 2,
}

WINDOWO_DATA_DIR = "data/window-open/"
WINDOW_OPEN = {
	"env_name": "window-open-v3",
	"img_dir": WINDOWO_DATA_DIR + IMG_DIR,
	"models": WINDOWO_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

CURRENT_ENV = BUTTON

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
PROP_SIZE = 8


N_ROUNDS = 400 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/1000
PPO_STEPS = 100000
DREAM_LEN = 30
PPO_LR = 0.0003
ACTION_REPEAT = True
INIT_GATHER = 5000