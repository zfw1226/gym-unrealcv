ENV_NAME = 'Search-RrDoorDiscrete-v0'

CONTINUE = False #load a pre-trained model
RESTART_EP = 6000 # the episode number of the pre-trained model

TRAIN = True # train the network
USE_TARGET_NETWORK = False # use the target network
SHOW = False # show the current state, reward and action
MAP = False # show the trajectory in 2d map

TF_DEVICE = '/gpu:0'
MAX_EPOCHS = 10000 # max episode number
MEMORY_SIZE = 50000
LEARN_START_STEP = 10000
INPUT_SIZE = 84
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # 1e6
GAMMA = 0.95
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
MAX_EXPLORE_STEPS = 50000
TEST_INTERVAL_EPOCHS = 100000
SAVE_INTERVAL_EPOCHS = 500

LOG_NAME_SAVE = 'log'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
TRA_DIR = LOG_NAME_SAVE + '/trajectory.csv' # the path to save trajectory

LOG_NAME_READ = 'log'
#the path to reload weights, monitor and params
weights_path = LOG_NAME_READ + '/model/dqn_ep' + str(RESTART_EP)+ '.h5'
monitor_path = LOG_NAME_READ + '/monit` or/'+ str(RESTART_EP)
params_json = LOG_NAME_READ + '/param/dqn_ep' + str(RESTART_EP) + '.json'

