FINAL_VERSION = True
DREAM_VERSION = False

IMG_DIR = "imgs/"
TRANSITIONS = "transition_data.json"
VAE_MODEL = "vae_model.pth"
MDRNN_MODEL = "mdrnn_model.pth"
PPO_MODEL = "ppo_model"

if FINAL_VERSION:
	VAE_MODEL = "final_models/" + VAE_MODEL
	if DREAM_VERSION:
		MDRNN_MODEL = "final_models/dream_env/mdrnn_model.pth"
		PPO_MODEL = "final_models/dream_env/ppo_model"
	else:
		MDRNN_MODEL = "final_models/real_env/mdrnn_model.pth"
		PPO_MODEL = "final_models/real_env/ppo_model"

PUSHER_DATA_DIR = "data/pusher/"
PUSHER = {
	"env_name": "Pusher-v5",
	"data_dir": PUSHER_DATA_DIR,
	"img_dir": PUSHER_DATA_DIR + IMG_DIR,
	"transitions": PUSHER_DATA_DIR + TRANSITIONS,
	"vae_model": PUSHER_DATA_DIR + VAE_MODEL,
	"mdrnn_model": PUSHER_DATA_DIR + MDRNN_MODEL,
	"ppo_model": PUSHER_DATA_DIR + PPO_MODEL,
	"z_size": 32,
	"rnn_size": 256,
	"num_gaussians": 5,
	"a_size": 7,
	"default_camera_config": {
		"trackbodyid": -1,   # no specific body tracking
		"distance": 1.8,     # distance from the agent
		"azimuth": -90.0,    # rotate camera to front of the agent
		"elevation": -20.0,  # lower angle for a front view
	},
	"special_call": None
}

def no_render_indicators():
	from gymnasium.envs.box2d import car_racing
	def no_render_(self, W, H):
		return
	car_racing.CarRacing._render_indicators = no_render_
CAR_RACING_DATA_DIR = "data/car_racing/"
CAR_RACING = {
	"env_name": "CarRacing-v3",
	"data_dir": CAR_RACING_DATA_DIR,
	"img_dir": CAR_RACING_DATA_DIR + IMG_DIR,
	"transitions": CAR_RACING_DATA_DIR + TRANSITIONS,
	"vae_model": CAR_RACING_DATA_DIR + VAE_MODEL,
	"mdrnn_model": CAR_RACING_DATA_DIR + MDRNN_MODEL,
	"ppo_model": CAR_RACING_DATA_DIR + PPO_MODEL,
	"z_size": 32,
	"rnn_size": 512,
	"num_gaussians": 7,
	"a_size": 3,
	"default_camera_config": None,
	"special_call": no_render_indicators
}

CURRENT_ENV = PUSHER