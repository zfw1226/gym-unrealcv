ENV_NAME = 'Unrealcv-Search-v5'

CONTINUE = True #load a pre-trained model
RESTART_EP = 3200 # the episode number of the pre-trained model

TRAIN = False # train the network
USE_TARGET_NETWORK = False # use the target network
SHOW = True # show the current state, reward and action
MAP = False # show the trajectory in 2d map

MAX_EPOCHS = 100000 # max episode number

MEMORY_SIZE = 10000
LEARN_START_STEP = 5000
INPUT_SIZE = 150
INPUT_CHANNELS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # 1e6
GAMMA = 0.95
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
MAX_EXPLORE_STEPS = 5000
TEST_INTERVAL_EPOCHS = 100000
SAVE_INTERVAL_EPOCHS = 200

LOG_NAME_SAVE = 'log-angle-v5.1-eval'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
TRA_DIR = LOG_NAME_SAVE + '/trajectory.csv' # the path to save trajectory


LOG_NAME_READ = 'log-angle-v4.0'
#the path to reload weights, monitor and params
weights_path = LOG_NAME_READ + '/model/dqn_ep' + str(RESTART_EP)+ '.h5'
monitor_path = LOG_NAME_READ + '/monitor/'+ str(RESTART_EP)
params_json = LOG_NAME_READ + '/param/dqn_ep' + str(RESTART_EP) + '.json'
