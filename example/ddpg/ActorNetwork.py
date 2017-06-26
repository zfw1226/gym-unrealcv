import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K


class ActorNetwork(object):
    def __init__(self, sess, state_shape, action_size, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_shape, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_shape, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_shape,action_size):
        print("Now we build the model")

        S = Input(shape= state_shape)

        c1 = Convolution2D(32, 3, 3, activation='relu')(S)
        c2 = Convolution2D(32, 3, 3, activation='relu')(c1)
        c3 = MaxPooling2D(pool_size=(2, 2))(c2)

        c4 = Convolution2D(32, 3, 3, activation='relu')(c3)
        c5 = Convolution2D(32, 3, 3, activation='relu')(c4)
        c6 = MaxPooling2D(pool_size=(2, 2))(c5)
        c7 = Flatten()(c6)
        c8 = Dense(512, activation='relu')(c7)
        c9 = Dense(256, activation='relu')(c8)
        output = Dense(action_size, activation = 'sigmoid')(c9)
        model = Model(input=S, output=output)
        model.summary()

        return model, model.trainable_weights, S

