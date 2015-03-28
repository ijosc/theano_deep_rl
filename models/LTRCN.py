import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import random

class Model(object):
    def __init__(self, input_width, input_height, number_of_actions,
                 batch_size=1, channels=4, discount_factor=0.99):
        self.input_width = input_width
        self.input_height = input_height
        self.channels = channels
        self.batch_size = batch_size
    
        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor

        self.l_out = self.build()

        state = T.tensor4()
        qvalues = T.vector()
        qvalues_reinforced = T.vector()

        output = self.l_out.get_output(state)
        self.predict = theano.function([state], output)

        objective = lasagne.objectives.Objective(
            self.l_out,
            loss_function=lasagne.objectives.mse)

        cost = objective.get_loss(state, target=qvalues_reinforced)

        all_params = lasagne.layers.get_all_params(self.l_out)

        updates = lasagne.updates.rmsprop(
            cost, 
            all_params, 
            learning_rate=0.0001, 
            rho=0.9, 
            epsilon=1e-06)

        self.train = theano.function(
            inputs=[state, qvalues_reinforced],
            outputs=cost,
            updates=updates)

    def build(self):
        l_in = lasagne.layers.InputLayer(
            shape=(
                self.batch_size,
                self.channels,
                self.input_width,
                self.input_height),
            )

        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            strides=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.0001))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            strides=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01))

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            strides=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01))

        l_dense = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.1))

        l_reshape = lasagne.layers.ReshapeLayer(
            l_dense,
            (self.batch_size,
             l_dense.get_output_shape()[0],
             l_dense.get_output_shape()[1]))
        print l_reshape.get_output_shape()
        l_LSTM1 = lasagne.layers.LSTMLayer(
            l_reshape,
            num_units=102,
            peepholes=True,
            learn_init=True)

        l_out = lasagne.layers.DenseLayer(
            l_LSTM1,
            num_units=self.number_of_actions,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(std=0.1))

        return l_out

    def train_step(self, pre_state, action,
                   reward, post_state, game_length):
        self.batch_size = game_length
        qvalues = self.predict(pre_state)[0]
        max_qvalue = np.max(self.predict(post_state))

        qvalues_reinforced = qvalues.copy()
        qvalues_reinforced[action] = reward + \
                                     self.discount_factor * max_qvalue

        self.train(pre_state, qvalues_reinforced)

    def greedy_step(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(self.number_of_actions))
        else:
            action = np.argmax(self.predict(state))

        return action

