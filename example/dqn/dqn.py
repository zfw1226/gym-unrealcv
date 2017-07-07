import time
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import memory
import keras.backend as K
class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, outputs, memorySize, discountFactor, learningRate, img_rows, img_cols, img_channels ,useTargetNetwork):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0
        if K.backend() == 'tensorflow':
            with KTF.tf.device('/gpu:1'):
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                KTF.set_session(tf.Session(config=config))
                self.initNetworks()
        else :
            self.initNetworks()

    def initNetworks(self):

        self.model = self.createModel()
        if self.useTargetNetwork:
            self.targetModel = self.createModel()

    def createModel(self):
        input_shape = (self.img_channels, self.img_rows, self.img_cols)
        if K.image_dim_ordering() == 'tf':
            input_shape = ( self.img_rows, self.img_cols, self.img_channels)

        model = Sequential()
        model.add(Convolution2D(32, 3, 3,border_mode='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())


        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.output_size,activation='linear'))
        model.compile(Adam(lr=self.learningRate), 'MSE')
        model.summary()


        return model


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
        print 'update target network'

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.targetModel.predict(state)
        else:
            predicted = self.model.predict(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize,):

        self.count_steps += 1

        state_batch,action_batch,reward_batch,newState_batch,isFinal_batch\
        = self.memory.getMiniBatch(miniBatchSize)

        qValues_batch = self.model.predict(np.array(state_batch),batch_size=miniBatchSize)

        isFinal_batch = np.array(isFinal_batch) + 0

        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel.predict_on_batch(np.array(newState_batch))
        else :
            qValuesNewState_batch = self.model.predict_on_batch(np.array(newState_batch))

        Y_sample_batch = reward_batch + (1 - isFinal_batch) * self.discountFactor * np.max(qValuesNewState_batch, axis=1)

        X_batch = np.array(state_batch)
        Y_batch = np.array(qValues_batch)

        for i,action in enumerate(action_batch):
            Y_batch[i][action] = Y_sample_batch[i]

        self.model.fit(X_batch, Y_batch, validation_split=0.0, batch_size = miniBatchSize, nb_epoch=1, verbose = 0)

        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()


    def saveModel(self, path):
        if self.useTargetNetwork:
            self.targetModel.save(path)
        else:
            self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())
        if self.useTargetNetwork:
            self.targetModel.set_weights(load_model(path).get_weights())


    def feedforward(self,observation,explorationRate):
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action






