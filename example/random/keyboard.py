from pynput import keyboard
import argparse
import gym_unrealcv
import gym



def on_press(key):
    try:
        #print('alphanumeric key {0} pressed'.format(key.char))
        if key.char == 'r':
            env.reset()
        elif key.char >= '0' and key.char < str(env.action_space.n):
            action = int(key.char)
            ob, reward, done, _ = env.step(action)
            print reward
            if done:
                env.reset()

    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='RobotArm-Discrete-v1', help='Select the environment to run')
args = parser.parse_args()
env = gym.make(args.env_id)
env.reset()
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# Close the env and write monitor result info to disk
env.close()