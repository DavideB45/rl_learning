EXP_ID = 7
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
	"camera_id": 2,
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

DRAWERO_DATA_DIR = "data/drawer-open/"
DRAWER_OPEN = {
	"env_name": "drawer-open-v3",
	"img_dir": DRAWERO_DATA_DIR + IMG_DIR,
	"models": DRAWERO_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 4,
	"render_size": 64,
	"camera_id": 2,
}

PEG_DATA_DIR = "data/peg-insert/"
PEG_INSERT = {
	"env_name": "peg-insert-side-v3",
	"img_dir": PEG_DATA_DIR + IMG_DIR,
	"models": PEG_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
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

LATENT_DIM = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 32
SMOOTH = 5
VQ_EPOCS = 40 # used in learning loop for the initial training, then 1 epoch for each round
VQ_LR = 1e-3
VQ_WD = 0.001

EP_ON_LOOP = 20

SEQ_LEN = 25 #12
INIT_LEN = 10
REW_WEIGHT = 1
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


N_ROUNDS = 1000 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/500
PPO_STEPS = 100000
DREAM_LEN = 30
PPO_LR = 0.0003
PPO_KL = 0.03
ACTION_REPEAT = True
INIT_GATHER = 5000


# ------------------------------------------------------------------
# SAC (agent trained inside the dream / world model)
# Paste these into global_var.py
# ------------------------------------------------------------------
DREAM_NUM_ENVS = 50           # parallel dream envs (was hard-coded to 50)

SAC_STEPS = 50_000           # env steps per round, summed over all dream envs
SAC_LR = 3e-4
SAC_BUFFER_SIZE = 300_000     # smaller buffer = fresher data w.r.t. the changing LSTM
SAC_BATCH_SIZE = 512
SAC_LEARNING_STARTS = 10_000  # random-action warm-up before the first update (first round only)
SAC_TAU = 0.005               # target network soft-update rate
SAC_GAMMA = 0.99
SAC_TRAIN_FREQ = 1            # 1 vectorised step = DREAM_NUM_ENVS transitions
SAC_GRADIENT_STEPS = 4        # updates per train_freq -> main compute/wall-clock knob
SAC_ENT_COEF = 'auto'         # learned temperature; or a float like 0.01
SAC_TARGET_ENTROPY = 'auto'   # -dim(action_space)
SAC_USE_SDE = True
SAC_SDE_SAMPLE_FREQ = -1

SAC_PI_ARCH = [1024, 512, 256]  # same size as the PPO nets, for a fair comparison
SAC_QF_ARCH = [1024, 512, 256]
SAC_N_CRITICS = 2

SAC_RESET_BUFFER_ON_VQ_UPDATE = True  # clear buffer when the VQ-VAE is retrained
SAC_REFILL_STEPS = 5_000              # warm-up steps after a buffer reset
SAC_SAVE_BUFFER = False               # save buffer to disk each round (can be large)