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

PUSH_DATA_DIR = "data/push/"
PUSH = {
	"env_name": "push-v3",
	"img_dir": PUSH_DATA_DIR + IMG_DIR,
	"models": PUSH_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

DRAWERO_DATA_DIR = "data/drawer-open/"
DRAWER_OPEN = {
	"env_name": "drawer-open-v3",
	"img_dir": DRAWERO_DATA_DIR + IMG_DIR,
	"models": DRAWERO_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

PEG_DATA_DIT = "data/peg-insert/"
PEG_INSERT = {
	"env_name": "peg-insert-side-v3",
	"img_dir": PEG_DATA_DIT + IMG_DIR,
	"models": PEG_DATA_DIT + MODELS_DIR,
	"a_size": 4,
	"camera_id": 2,
}

HAMMER_DATA_DIR = "data/hammer/"
HAMMER = {
	"env_name": "hammer-v3",
	"img_dir": HAMMER_DATA_DIR + IMG_DIR,
	"models": HAMMER_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

PICKB_DATA_DIR = "data/bin-pick/"
PICK_BIN = {
	"env_name": "bin-picking-v3",
	"img_dir": PICKB_DATA_DIR + IMG_DIR,
	"models": PICKB_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

PICKP_DATA_DIR = "data/pick-place/"
PICK_PLACE = {
	"env_name": "pick-place-v3",
	"img_dir": PICKP_DATA_DIR + IMG_DIR,
	"models": PICKP_DATA_DIR + MODELS_DIR,
	"a_size": 4,
	"camera_id": 1,
}

CURRENT_ENV = PICK_PLACE

LATENT_DIM = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 64
SMOOTH = 5
VQ_EPOCS = 20 # used in learning loop for the initial training, then 1 epoch for each round
VQ_LR = 1e-3
VQ_WD = 0.001

EP_ON_LOOP = 20

SEQ_LEN = 25
INIT_LEN = 10
USE_KL = True

HIDDEN_DIM = 1024
LSTM_EPOCS = 2 # used in learning loop for the initial training, then 1 epoch for each round
LSTM_LR = 5e-5
LSTM_WD = 1e-3

TR_EPOCHS = 10
TR_LR = 1e-4
TR_WD = 1e-3
EMB_SIZE = 128
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = INIT_LEN + 1
DROPOUT = 0.0


N_ROUNDS = 990 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/1000
PPO_STEPS = 200000
DREAM_LEN = 30
PPO_LR = 0.0003
ACTION_REPEAT = True
INIT_GATHER = 5000