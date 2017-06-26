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

class ActorCriticNetwork(object):
    def __init__(self, sess, state_shape, action_size, TAU, LEARNING_RATE, target_size):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)




        self.model_A , self.weights_A, self.state_A = self.create_actor_network(state_shape, action_size, trainable = True)
        self.target_model_A, self.target_weights_A, self.target_state_A = self.create_actor_network(state_shape, action_size, trainable = True)

        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model_A.output, self.weights_A, -self.action_gradient)
        grads = zip(self.params_grad, self.weights_A)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads, targets):
        self.sess.run(self.optimize, feed_dict={
            self.state_A: states,
            self.action_gradient: action_grads,
        })

    def gradients_critic(self, states, actions, targets):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions,
        })[0]

    def train_actor(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
        })


    def target_train(self, model, target_model):
        actor_weights = model.get_weights()
        actor_target_weights = target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        target_model.set_weights(actor_target_weights)

    def perception_network(self,state_shape,trainable = True):
        S = Input(shape=state_shape)

        c1 = Convolution2D(16, 3, 3, activation='relu', trainable = trainable)(S)
        c2 = Convolution2D(16, 3, 3, activation='relu', trainable = trainable)(c1)
        c3 = MaxPooling2D(pool_size=(2, 2))(c2)

        c4 = Convolution2D(32, 3, 3, activation='relu', trainable = trainable)(c3)
        c5 = Convolution2D(32, 3, 3, activation='relu', trainable = trainable)(c4)
        c6 = MaxPooling2D(pool_size=(2, 2))(c5)
        c7 = Flatten()(c6)
        return c7, S

    def actor_network(self, input_tensor, action_size, trainable = True):
        c8 = Dense(512, activation='relu', trainable = trainable)(input_tensor)
        c9 = Dense(256, activation='relu', trainable = trainable)(c8)
        action = Dense(action_size, activation = 'sigmoid',name = 'action', trainable = trainable)(c9)
        return action

    def critic_network(self,input_tensor, action_size, merge_dim, trainable = True):
        A = Input(shape=[action_size],name='action', trainable = trainable)
        a1 = Dense(merge_dim, activation='linear', trainable = trainable)(A)
        c = Dense(merge_dim, activation='relu', trainable = trainable)(input_tensor)

        h = merge([c,a1],mode='sum')
        h = Dense(256, activation='relu', trainable = trainable)(h)
        V = Dense(action_size,activation='linear', name='value',trainable = trainable)(h)
        return V,A

    def create_network(self,state_shape, action_size, p_train, a_train, c_train ):
        perception, state = self.perception_network(state_shape,trainable=p_train)
        action = self.action_network(perception, action_size, a_train)
        value = self.critic_network(perception, action_size, 512, c_train)
        model = Model(input=[state], output=[action,value])
        model.summary()
        return model, model.trainable_weights, state


    def create_actor_network(self, state_shape,action_size, trainable = True):
        print("Now we build the actor model")
        perception, state = self.perception_network(state_shape, trainable = trainable)
        action = self.action_network(perception, action_size, trainable = trainable)
        model = Model(input=perception, output=action)
        model.summary()

        return model, model.trainable_weights, perception

    def create_critic_network(self, state_shape,action_size, trainable = True):
        print("Now we build the critic model")

        perception, state = self.perception_network(state_shape, trainable = trainable)
        value, action = self.critic_network(perception, action_size, 512, trainable = trainable)
        model = Model(input=perception, output=value)
        model.summary()

        return model, action, state