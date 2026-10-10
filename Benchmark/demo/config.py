'''
Settings for the lab demo (school visit). Edit and re-run, there are no CLI flags for these.

The demo does NOT use any model: it replays the pressures recorded in the dataset of one
experiment. In the dataset json the first INIT_EPISODES episodes are the initial random data
gathering, then episode INIT_EPISODES + r - 1 is the one collected after training round r.
'''
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

EXP_ID = 101
DATA_DIR = os.path.join(REPO_ROOT, f'data/real-soft/imgs/tr/round_{EXP_ID}')
TRANSITIONS = 'action_reward_data.json'
INIT_EPISODES = 10


def round_to_episode(r):
    '''training round (1-based, as in the csv log) -> episode index in the json'''
    return INIT_EPISODES + r - 1


# the three real tries shown to the kids (training round, as in res_<EXP_ID>.csv)
TRY_ROUNDS = [1, 20, 90]

# the "dreams" between the tries: DREAM_CLIPS episodes picked from these round ranges
# (inclusive), shown sorted from worst to best so the dream visibly gets better
DREAM_ROUND_RANGES = [(2, 19), (21, 89)]
DREAM_CLIPS = 2
DREAM_FPS = 30            # recorded at 10 Hz, dreams play 3x faster
DREAM_CLIP_PAUSE = 1.0    # seconds of pause after each dream clip (shows its score)

# timings (seconds)
INTRO_TIME = 6.0          # "Tentativo N" card before each real try
RESULT_TIME = 7.0         # result card after each real try (robot deflates meanwhile)
REPLAY_HZ = 10            # same rate as the RealWorld env (env_hz)
HOLD_AFTER_REPLAY = 1.0   # keep the last pressure a moment before deflating
# if True, intro/result cards wait for the "Avanti" button (or space bar) instead of the timer
WAIT_FOR_CLICK = False

# reward = rotation [rad] * REW_MULTIPLIER in the dataset (same as envs/wrapper.py)
REW_MULTIPLIER = 30.0
GOAL_DEG = 180.0          # success threshold used by RealWorld (half a turn)

# hardware (same values as envs/wrapper.py)
# the top camera looks at the wheel: it's owned by the ArUco tracker, which measures the rotation
# and also provides the main video (only frames where the marker is visible are streamed)
TOP_CAMERA_ID = 0
ARUCO_MARKER_ID = 5
ARUCO_MIN_SHARPNESS = 100.0  # same as the env; lower it (or None) if the video looks too choppy
# optional front camera (the one used for the dataset images): None = not connected.
# If set, it becomes the main video and the top camera goes in the small round inset.
FRONT_CAMERA_ID = None
CAMERA_SIZE = 480
MAX_PRESSURE = 0.8

PORT = 8000
