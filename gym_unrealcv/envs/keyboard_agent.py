from pynput import keyboard
import argparse
import gym_unrealcv
import gym
from queue import Queue

queue = Queue()
global keys
discrete_actions=[
    [ 200,  0], # move forward
    [-200,  0], # move backward
    [ 100, 20], # turn clockwise
    [ 100,-20], # turn anticlockwise
    [  0,   0]  # stay
]

# use keyboard "1,2,3,4" and "i,j,k,l" to control two players sepreately
# 1 & i: move forward
# 2 & k: move backward
# 3 & j: turn anticlockwise
# 4 & l: turn clockwise
def on_press(key):
    try:
        if key.char == 'r':
            env.reset()
        elif key.char in ['i','j','k','l'] or key.char in ['1','2','3','4'] :
            queue.put(key.char)
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrackMulti-FlexibleRoomAdv-DiscreteColor-v2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.seed(int(args.seed))
    obs = env.reset()


    keyboard_thread=keyboard.Listener(on_press=on_press,on_release=on_release)
    keyboard_thread.start()
    print('start press button')

    while keyboard_thread.is_alive():
        if env.env.action_type =="Discrete":
            action_id = queue.get()
            if action_id == '1':
                actions = [discrete_actions[0],discrete_actions[4]]
            elif action_id == '2':
                actions = [discrete_actions[1],discrete_actions[4]]
            elif action_id == '3':
                actions = [discrete_actions[3],discrete_actions[4]]
            elif action_id == '4':
                actions = [discrete_actions[2],discrete_actions[4]]
            elif action_id == 'i':
                actions = [discrete_actions[4],discrete_actions[0]]
            elif action_id == 'j':
                actions = [discrete_actions[4],discrete_actions[3]]
            elif action_id == 'k':
                actions = [discrete_actions[4],discrete_actions[1]]
            elif action_id == 'l':
                actions = [discrete_actions[4],discrete_actions[2]]

        env.unwrapped.unrealcv.set_move_batch(env.env.player_list,actions)

    # Close the env and write monitor result info to disk
    env.close()