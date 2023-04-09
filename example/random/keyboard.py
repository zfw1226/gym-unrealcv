from pynput import keyboard
import argparse
import gym_unrealcv
import gym

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='RobotArm-Discrete-v1', help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()

if args.render:
    import cv2

def on_press(key):
    try:
        #print('alphanumeric key {0} pressed'.format(key.char))
        if key.char == 'r':
            env.reset()
        elif key.char >= '0' and key.char < str(env.action_space.n):  # for discrete action
            action = int(key.char)
            ob, reward, done, _ = env.step(action)
            print(reward)
            if done:
                print('Reset!')
                env.reset()
            if args.render:
                img = env.render(mode='rgb_array')
                cv2.imshow('show', img)
                cv2.waitKey(1)

    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

env = gym.make(args.env_id)
env.reset()
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

env.close() # Close the env