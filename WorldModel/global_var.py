IMG_DIR = "img/"

PENDULUM_DATA_DIR = "data/pendulum/"
PENDULUM = {
	"env_name": "Pendulum-v1",
	"data_dir": PENDULUM_DATA_DIR,
	"img_dir": PENDULUM_DATA_DIR + IMG_DIR,
	"default_camera_config": None,
	"special_call": None
}

PUSHER_DATA_DIR = "data/pusher/"
PUSHER = {
	"env_name": "Pusher-v5",
	"data_dir": PUSHER_DATA_DIR,
	"img_dir": PUSHER_DATA_DIR + IMG_DIR,
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
	"default_camera_config": None,
	"special_call": no_render_indicators
}

CURRENT_ENV = CAR_RACING