import gym
import gym_unrealcv
from distutils.dir_util import copy_tree
import os
import json
from constants import *
from ddpg import DDPG
from gym import wrappers
import time
from example.utils import preprocessing, io_util


if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env.rendering = SHOW
    assert env.action_type == 'continuous'
    ACTION_SIZE = env.action_space.shape[0]
    ACTION_HIGH = env.action_space.high
    ACTION_LOW = env.action_space.low
    INPUT_CHANNELS = env.observation_space.shape[2]
    OBS_HIGH = env.observation_space.high
    OBS_LOW = env.observation_space.low
    OBS_RANGE = OBS_HIGH - OBS_LOW

    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 3, size = (INPUT_SIZE,INPUT_SIZE))

    #init log file
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(PARAM_DIR):
        os.makedirs(PARAM_DIR)

    Agent = DDPG(ACTION_SIZE, MEMORY_SIZE, GAMMA,
                 LEARNINGRATE_CRITIC, LEARNINGRATE_ACTOR, TARGET_UPDATE_RATE,
                 INPUT_SIZE, INPUT_SIZE, 3)
    #load init param
    if not CONTINUE:
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0
        env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,force=True)

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            explorationRate = d.get('explorationRate')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            Agent.loadWeights(critic_weights_path, actor_weights_path)
            io_util.clear_monitor_files(MONITOR_DIR + 'tmp')
            copy_tree(monitor_path, MONITOR_DIR + 'tmp')
            env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,resume=True)

    if not os.path.exists(TRA_DIR):
        io_util.create_csv_header(TRA_DIR)

    try:
        start_time = time.time()
        for epoch in xrange(current_epoch + 1, MAX_EPOCHS + 1, 1):
            obs = env.reset()
            #observation = io_util.preprocess_img(obs)
            observation = process_img.process_gray(obs,reset=True)
            cumulated_reward = 0
            #if ((epoch) % TEST_INTERVAL_EPOCHS != 0 or stepCounter < LEARN_START_STEP) and TRAIN is True :  # explore
            EXPLORE = True
            #else:
            #    EXPLORE = False
            #    print ("Evaluate Model")
            for t in xrange(MAX_STEPS_PER_EPOCH):

                start_req = time.time()

                if EXPLORE is True: #explore

                    action_pred = Agent.actor.model.predict(observation)
                    action = Agent.Action_Noise(action_pred, explorationRate)
                    #print action

                    action_env = action * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
                    obs_new, reward, done, info = env.step(action_env)

                    newObservation = process_img.process_gray(obs_new)
                    #newObservation = io_util.preprocess_img(obs_new)
                    stepCounter += 1

                    Agent.addMemory(observation, action, reward, newObservation, done)
                    observation = newObservation
                    if stepCounter == LEARN_START_STEP:
                        print("Starting learning")


                    if Agent.getMemorySize() >= LEARN_START_STEP:
                        Agent.learnOnMiniBatch(BATCH_SIZE)
                        if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EXPLORE_STEPS
                        #elif stepCounter % (MAX_EXPLORE_STEPS * 1.5) == 0:
                        #    explorationRate = 0.99
                        #    print 'Reset Exploration Rate'
                #test
                else:
                    action = Agent.actor.model.predict(observation)
                    obs_new, reward, done, info = env.step(action)
                    newObservation = process_img.process_gray(obs_new)
                    #newObservation = io_util.preprocess_img(obs_new)
                    observation = newObservation

                #print 'step time:' + str(time.time() - start_req)
                if MAP:
                    io_util.live_plot(info)
                #io_util.save_trajectory(info, TRA_DIR, epoch)

                cumulated_reward += reward
                if done:
                    m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                    h, m = divmod(m, 60)

                    print ("EP " + str(epoch) +" Csteps= " + str(stepCounter) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s) )
                        # SAVE SIMULATION DATA
                    if (epoch) % SAVE_INTERVAL_EPOCHS == 0 and TRAIN is True:
                        # save model weights and monitoring data
                        print 'Save model'
                        Agent.saveModel( MODEL_DIR + '/ep' +str(epoch))

                        copy_tree(MONITOR_DIR + 'tmp', MONITOR_DIR + str(epoch))
                        # save simulation parameters.
                        parameter_keys = ['explorationRate', 'current_epoch','stepCounter', 'FINAL_EPSILON','loadsim_seconds']
                        parameter_values = [explorationRate, epoch, stepCounter,FINAL_EPSILON, int(time.time() - start_time + loadsim_seconds)]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(PARAM_DIR + '/' + str(epoch) + '.json','w') as outfile:
                            json.dump(parameter_dictionary, outfile)

                    break

    except KeyboardInterrupt:
        print("Shutting down")
        env.close()
