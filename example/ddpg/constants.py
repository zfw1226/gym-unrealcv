ENV_NAME = 'Search-Arch1Test-v0'

CONTINUE = False
RESTART_EP = 8000
TRAIN = True
SHOW = True
MAP = False

MAX_EPOCHS = 100000
MAX_STEPS_PER_EPOCH = 10000
MEMORY_SIZE = 10000
LEARN_START_STEP = 5000
INPUT_SIZE = 150 # pre 100
INPUT_CHANNELS = 3
BATCH_SIZE = 64

LEARNINGRATE_CRITIC  = 0.001
LEARNINGRATE_ACTOR = 0.0001
TARGET_UPDATE_RATE = 0.001

GAMMA = 0.95
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon

MAX_EXPLORE_STEPS = 10000
TEST_INTERVAL_EPOCHS = 1000
SAVE_INTERVAL_EPOCHS = 200

LOG_NAME_SAVE = 'log-test'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
TRA_DIR = LOG_NAME_SAVE + '/trajectory.csv' # the path to save trajectory

#the path to reload weights, monitor and params
LOG_NAME_READ = 'log-angle-v4.0'
critic_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'Critic_model.h5'
actor_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'Actor_model.h5'
monitor_path = LOG_NAME_READ + '/monitor/'+ str(RESTART_EP)
params_json = LOG_NAME_READ + '/param/' + str(RESTART_EP) + '.json'