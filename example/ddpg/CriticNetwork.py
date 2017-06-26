import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation,Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_shape, action_size, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_shape, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_shape, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_shape,action_size):
        print("Now we build the model")
        S = Input(shape= state_shape)
        A = Input(shape=[action_size],name='action')
        a1 = Dense(512, activation='linear')(A)
        c1 = Convolution2D(32, 3, 3, activation='relu')(S)
        c2 = Convolution2D(32, 3, 3, activation='relu')(c1)
        c3 = MaxPooling2D(pool_size=(2, 2))(c2)

        c4 = Convolution2D(32, 3, 3, activation='relu')(c3)
        c5 = Convolution2D(32, 3, 3, activation='relu')(c4)
        c6 = MaxPooling2D(pool_size=(2, 2))(c5)
        c7 = Flatten()(c6)
        c8 = Dense(512, activation='relu')(c7)
        h2 = merge([c8,a1],mode='sum')
        h3 = Dense(256, activation='relu')(h2)
        V = Dense(self.action_size,activation='linear')(h3)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, A, S

