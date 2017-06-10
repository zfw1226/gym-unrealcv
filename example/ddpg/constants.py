ENV_NAME = 'Unrealcv-Search-v1'

CONTINUE = False
RESTART_EP = 4000
TRAIN = True
SHOW = True
MAP = False

MAX_EPOCHS = 10000
MAX_STEPS_PER_EPOCH = 100
MEMORY_SIZE = 20000
LEARN_START_STEP = 5000
INPUT_SIZE = 100
INPUT_CHANNELS = 3
BATCH_SIZE = 64
VELOCITY_MAX = 100
ANGLE_MAX = 45

LEARNINGRATE_CRITIC  = 0.001
LEARNINGRATE_ACTOR = 0.0001
TARGET_UPDATE_RATE = 0.001

GAMMA = 0.95
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon

MAX_EXPLORE_STEPS = 20000
TEST_INTERVAL_EPOCHS = 1000
SAVE_INTERVAL_EPOCHS = 200


MONITOR_DIR = 'log/monitor/' #the path to save monitor file
MODEL_DIR = 'log/model' # the path to save deep model
PARAM_DIR = 'log/param' # the path to save the parameters
TRA_DIR = 'log/trajectory.csv' # the path to save trajectory

#the path to reload weights, monitor and params
critic_weights_path = 'log/model/ep' + str(RESTART_EP) + '/actormodel.h5'
actor_weights_path = 'log/model/ep' + str(RESTART_EP) + '/criticmodel.h5'
monitor_path = 'log/monitor/'+ str(RESTART_EP)
params_json = 'log/param/' + str(RESTART_EP) + '.json'