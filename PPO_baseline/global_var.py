EXP_ID = 6
LOG_NAME = f'res_{EXP_ID}'
GPU_ID = f"{EXP_ID%4}" # window 2 - peg 3

IMG_DIR = "imgs/"
TRANSITIONS = "action_reward_data.json"
MODELS_DIR = "models/"


BUTTON_DATA_DIR = "data/button-press/"
BUTTON = {
	"env_name": "button-press-v3",
	"img_dir": BUTTON_DATA_DIR + IMG_DIR,
	"models": BUTTON_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 1,
}

BUTTON_TD_DATA_DIR = "data/button-press-td/"
BUTTON_TD = {
	"env_name": "button-press-topdown-v3",
	"img_dir": BUTTON_TD_DATA_DIR + IMG_DIR,
	"models": BUTTON_TD_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

PUSH_DATA_DIR = "data/push/"
PUSH = {
	"env_name": "push-v3",
	"img_dir": PUSH_DATA_DIR + IMG_DIR,
	"models": PUSH_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

DRAWERO_DATA_DIR = "data/drawer-open/"
DRAWER_OPEN = {
	"env_name": "drawer-open-v3",
	"img_dir": DRAWERO_DATA_DIR + IMG_DIR,
	"models": DRAWERO_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
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
	"models": HAMMER_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

PICKB_DATA_DIR = "data/bin-pick/"
PICK_BIN = {
	"env_name": "bin-picking-v3",
	"img_dir": PICKB_DATA_DIR + IMG_DIR,
	"models": PICKB_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

PICKP_DATA_DIR = "data/pick-place/"
PICK_PLACE = {
	"env_name": "pick-place-v3",
	"img_dir": PICKP_DATA_DIR + IMG_DIR,
	"models": PICKP_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

WINDOWO_DATA_DIR = "data/window-open/"
WINDOW_OPEN = {
	"env_name": "window-open-v3",
	"img_dir": WINDOWO_DATA_DIR + IMG_DIR,
	"models": WINDOWO_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

CURRENT_ENV = DRAWER_OPEN


N_ROUNDS = 1000 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/1000
PPO_LR = 0.0003
PPO_MIN_LR = 1e-5
N_ENVS = 4
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 128
PPO_N_EPOCHS = 10
PPO_ENT_COEF = 0.005
NORMALIZE_REWARD = True
USE_IMPALA = False
PPO_FEATURES_DIM = 256
IMPALA_DEPTHS = (16, 32, 32)
ACTION_REPEAT = True
ACTION_REPEAT_STEPS = 2
FRAME_STACK = 3
GRAYSCALE = True
CHANNELS_FIRST = True