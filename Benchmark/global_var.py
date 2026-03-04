IMG_DIR = "imgs/"
TRANSITIONS = "transition_data.json"
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

LATENT_DIM = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 64
SMOOTH = 1
VQ_EPOCS = 50
VQ_LR = 1e-3
VQ_WD = 0.001
