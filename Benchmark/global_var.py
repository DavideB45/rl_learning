EXP_ID = 6
LOG_NAME = f'res_{EXP_ID}'
GPU_ID = f"{EXP_ID%4}" # window 2 - peg 3

IMG_DIR = "imgs/"
TRANSITIONS = "action_reward_data.json"
MODELS_DIR = "models/"

# ELEANOR trunk grasp task (MuJoCo), from the eleanor-sim repo (../../eleanor-sim, `gym_tasks.grasp_env.GraspEnv`)
# 15-step episodes, action = change of the 7 tendon actuations, proprioception = the 7 actuations
GRASP_DATA_DIR = "data/grasp/"
GRASP = {
	"env_name": "EleanorGrasp-v0",
	"img_dir": GRASP_DATA_DIR + IMG_DIR,
	"models": GRASP_DATA_DIR + MODELS_DIR + f"{EXP_ID}/",
	"a_size": 7,
	"render_size": 64,
	"camera": "close", # a view of eleanor_sim/configs/cameras.yaml: front, close, side, three_quarter
	"yaw_range": (-30.0, 30.0), # start pose turn [deg], the default training setup of the eleanor-sim notes
	"randomize_defects": False,
}

CURRENT_ENV = GRASP

LATENT_DIM = 4
CODE_DEPTH = 16
CODEBOOK_SIZE = 32
SMOOTH = 5
VQ_EPOCS = 20 # used in learning loop for the initial training, then 1 epoch for each round
VQ_LR = 1e-3
VQ_WD = 0.001

EP_ON_LOOP = 20 # the VQ-VAE trains on at most EP_ON_LOOP*500 images per round

# an episode of the grasp task gives exactly one SEQ_LEN window, so the LSTM datasets are sized in episodes
SEQ_EP_TR = 500 # episodes sampled for the LSTM training set each round
SEQ_EP_VL = 100 # episodes for the LSTM validation set (also the start states of the dream env)

SEQ_LEN = 15 # = GraspEnv.T, a whole episode
INIT_LEN = 1
REW_WEIGHT = 1
USE_KL = True

HIDDEN_DIM = 1024
LSTM_EPOCS = 2 # used in learning loop for the initial training, then 1 epoch for each round
LSTM_LR = 5e-5
LSTM_WD = 1e-3
PROP_SIZE = 7 # the tendon actuations; the object position of the GraspEnv observation is privileged and unused


N_ROUNDS = 1000 # starts with INIT_GATHER interacitons, then add 500 each round, N_rounds=(total_interactions-INIT_GATHER*2)/1000
PPO_STEPS = 20000
DREAM_LEN = SEQ_LEN - INIT_LEN
PPO_LR = 0.0003
INIT_GATHER = 5000