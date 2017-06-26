import gym
import gym_unrealcv
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
import dqn
import cv2
from constants import *
import io_util
from gym import wrappers
import csv





if __name__ == '__main__':


    env = gym.make(ENV_NAME)

    ACTION_SIZE = len(ACTION_LIST)
    ANGLE_SIZE = 4

    #init log file
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(PARAM_DIR):
        os.makedirs(PARAM_DIR)


    #load init param
    if not CONTINUE:
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0
        Agent = dqn.DeepQ(ACTION_SIZE,
                          MEMORY_SIZE,
                          GAMMA,
                          LEARNING_RATE,
                          INPUT_SIZE,
                          INPUT_SIZE,
                          INPUT_CHANNELS,
                          USE_TARGET_NETWORK,
                          ANGLE_SIZE,
                          LOSS_DIR
                          )
        env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,force=True)

        #io_util.create_csv_header(TRA_DIR)

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            '''explorationRate = d.get('explorationRate')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')'''
            explorationRate = INITIAL_EPSILON
            current_epoch = 0
            stepCounter = 0
            loadsim_seconds = 0

            Agent = dqn.DeepQ(
                ACTION_SIZE,
                MEMORY_SIZE,
                GAMMA,
                LEARNING_RATE,
                INPUT_SIZE,
                INPUT_SIZE,
                INPUT_CHANNELS,
                USE_TARGET_NETWORK,
                ANGLE_SIZE,
                LOSS_DIR
            )
            Agent.loadWeights(weights_path)
            io_util.clear_monitor_files(MONITOR_DIR + 'tmp')
            copy_tree(monitor_path, MONITOR_DIR + 'tmp')
            env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,resume=True)

        #io_util.create_csv_header(TRA_DIR)
    if not os.path.exists(TRA_DIR):
        io_util.create_csv_header(TRA_DIR)

    dir_loss = []
    #main loop
    try:
        start_time = time.time()
        for epoch in xrange(current_epoch, MAX_EPOCHS, 1):
            obs = env.reset()
            observation = io_util.preprocess_img(obs)
            cumulated_reward = 0
            direction_linear = 0

            if ((epoch) % TEST_INTERVAL_EPOCHS != 0 or stepCounter < LEARN_START_STEP) and TRAIN is True :  # explore
                EXPLORE = True
            else:
                EXPLORE = False
                print ("Evaluate Model")
            for t in xrange(1000):

                start_req = time.time()

                if EXPLORE is True: #explore
                    [action,direction_pre] = Agent.feedforward(observation, explorationRate)

                    dir_loss.append(abs(direction_pre - direction_linear))
                    dir_loss_mean = np.array(dir_loss[-min(100,len(dir_loss)):]).mean()

                    obs_new, reward, done, info = env.step(ACTION_LIST[action])
                    newObservation = io_util.preprocess_img(obs_new)
                    stepCounter += 1

                    #angle_onehot, angle_id = io_util.onehot_angle(info['Direction'],ANGLE_SIZE)

                    #direction_linear = ((info['Direction']+180)%360)/360.0
                    direction_linear = (info['Pose'][-1]%360) / 360.0

                    Agent.addMemory_new(observation, action, reward, newObservation, done, direction_linear )
                    observation = newObservation
                    if stepCounter == LEARN_START_STEP:
                        print("Starting learning")

                    if Agent.getMemorySize() >= LEARN_START_STEP:
                        Agent.learnOnMiniBatch(BATCH_SIZE)

                        if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EXPLORE_STEPS
                        #elif stepCounter%(MAX_EXPLORE_STEPS * 1.5) == 0 :
                            #explorationRate = 0.99
                            #print 'Reset Exploration Rate'

                #test
                else:
                    [action, direction_pre] = Agent.feedforward(observation,0)

                    obs_new, reward, done, info = env.step(ACTION_LIST[action])
                    newObservation = io_util.preprocess_img(obs_new)
                    observation = newObservation

                if SHOW:
                    io_util.show_info(info, obs_new, direction_pre)
                if MAP:
                    io_util.live_plot(info)

                io_util.save_trajectory(info,TRA_DIR,epoch)

                cumulated_reward += reward
                if done:
                    m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                    h, m = divmod(m, 60)

                    print ("EP " + str(epoch) +" Csteps= " + str(stepCounter) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s) + " Direction Loss: " + str(dir_loss_mean) )
                        # SAVE SIMULATION DATA
                    if (epoch) % SAVE_INTERVAL_EPOCHS == 0 and TRAIN is True:
                        # save model weights and monitoring data
                        print 'Save model'
                        Agent.saveModel(MODEL_DIR + '/dqn_ep' + str(epoch) + '.h5')

                        #backup monitor file
                        copy_tree(MONITOR_DIR+ 'tmp', MONITOR_DIR + str(epoch))

                        parameter_keys = ['explorationRate', 'current_epoch','stepCounter', 'FINAL_EPSILON','loadsim_seconds']
                        parameter_values = [explorationRate, epoch, stepCounter,FINAL_EPSILON, int(time.time() - start_time + loadsim_seconds)]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(PARAM_DIR + '/dqn_ep' + str(epoch) + '.json','w') as outfile:
                            json.dump(parameter_dictionary, outfile)



                    break



    except KeyboardInterrupt:
        print("Shutting down")
        #cv2.destroyAllWindows()
        #env.monitor.close() # not needed in latest gym update
        #env.close_docker()
        env.close()
