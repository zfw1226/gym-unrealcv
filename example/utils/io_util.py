
import csv
import os
import cv2
import keras.backend as K
from example.dqn.constants import *
import matplotlib.pyplot as plt
import numpy as np

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

def show_info( info):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv_img = info['Color']
    cv2.putText(cv_img, 'Reward:' + str(round(info['Reward'],3)), (200, 450), font, 0.5, (255, 255, 255), 2)
    #cv2.putText(cv_img, 'Velocity:' + str(info['Action'][0]), (500, 430), font, 0.5, (255, 255, 255), 2)
    #cv2.putText(cv_img, 'Angle:' + str(info['Action'][1]), (500, 450), font, 0.5, (255, 255, 255), 2)

    if info['Trigger']:
        cv2.putText(cv_img, 'Trigger', (400, 450), font, 0.5, (0, 0, 255), 2)
        cv2.circle(cv_img, (420, 420), 15, (0, 0, 255), -1)
    else:
        cv2.putText(cv_img, 'Trigger', (400, 450), font, 0.5, (255, 255, 255), 2)
        cv2.circle(cv_img, (420, 420), 15, (255, 255, 255), -1)

    if info['Collision']:
        cv2.circle(cv_img, (120, 420), 15, (0, 0, 255), -1)
        cv2.putText(cv_img, 'Collision', (100, 450), font, 0.5, (0, 0, 255), 2)
    else:
        cv2.circle(cv_img, (120, 420), 15, (0, 255, 0), -1)
        cv2.putText(cv_img, 'Collision', (100, 450), font, 0.5, (0, 255, 0), 2)

    for box in info['Bbox']:
            xmin = int(box[0][0] * cv_img.shape[1])
            xmax = int(box[1][0] * cv_img.shape[1])
            ymin = int(box[0][1] * cv_img.shape[0])
            ymax = int(box[1][1] * cv_img.shape[0])
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (50, 255, 50))

    cv2.imshow('info_show', cv_img)
    cv2.waitKey(3)

def save_trajectory(info,filedir,epoch):#save when every step finished
    trajec = dict()
    with open(filedir, 'a') as f:
        header = ['epoch', 'step', 'x', 'y', 'z', 'yaw','reward','collision','done']
        f_csv = csv.DictWriter(f, header)
    # list to dic and append dic to list
        trajec['x'],trajec['y'],trajec['z'],trajec['yaw'] = info['Trajectory'][-1]
        trajec['epoch'] = epoch
        trajec['step'] = info['Steps']
        trajec['reward'] = info['Reward']
        trajec['collision'] = info['Collision']
        trajec['done'] = info['Done']
        f_csv.writerows([trajec])

def create_csv_header(DIR):
    with open(DIR, 'a') as f:
        header = ['epoch', 'step', 'x', 'y', 'z', 'yaw','reward','collision','done']
        f_csv = csv.DictWriter(f, header)
        f_csv.writeheader()

def preprocess_img(image):
    cv_image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img_processed = cv_image.reshape(1, cv_image.shape[-1], INPUT_SIZE, INPUT_SIZE)
    #img_processed = img_processed / 255.0
    if K.image_dim_ordering() == 'tf':
        img_processed = img_processed.transpose(0, 2, 3, 1)
    return img_processed

def live_plot(info):
    plt.ion()
    X = []
    Y = []
    global line
    plt.xlim((-200, 200))
    plt.ylim((-550, 350))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(info['Target'][0], -info['Target'][1], c='red', s=150,  alpha=0.6, edgecolors='white',label='target object')
    #print info['Trajectory']
    if len(info['Trajectory']) > 0:
        for pos in info['Trajectory']:
            #print pos
            X.append(pos[0])
            Y.append(-pos[1])

        if info['Steps'] == 0:
            line = plt.plot(X, Y, 'b')
            plt.scatter(X[0], Y[0], c='blue', s=50, alpha=0.6, edgecolors='white', label='current path')  # plot start point

        if info['Done']:
            line[-1].set_xdata(X)
            line[-1].set_ydata(Y)
            line[-1].set_linestyle('--')

            if info['Reward'] > 0:
                line[-1].set_color('r')
                plt.scatter(X[-1], Y[-1], c='green', s=info['Reward']*20, alpha=0.6, edgecolors='white')# right trigger
            elif info['Collision']:
                line[-1].set_color('k')
                plt.scatter(X[-1], Y[-1], c='magenta', s=20, alpha=0.6, edgecolors='white')

        else:
            line[-1].set_xdata(X)
            line[-1].set_ydata(Y)
            line[-1].set_color('b')

        plt.pause(0.01)

def onehot(num, len):
    onehot = np.zeros(len)
    onehot[num] = 1
    return onehot
