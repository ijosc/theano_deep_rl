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
                 batch_size=500, channels=4, discount_factor=0.99):
        self.input_width = input_width
        self.input_height = input_height
        self.channels = channels
        self.batch_size = batch_size

        self.states = np.zeros((batch_size,
                                channels,
                                input_width,
                                input_height))

        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor

        self.l_out = self.build()

        state = T.tensor4()
        qvalues = T.vector()
        qvalues_reinforced = T.matrix()

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

        l_reshape_1 = lasagne.layers.ReshapeLayer(
            l_dense,
            (1,
             l_dense.get_output_shape()[0],
             l_dense.get_output_shape()[1]))

        l_LSTM1 = lasagne.layers.LSTMLayer(
            l_reshape_1,
            num_units=102,
            peepholes=True,
            learn_init=True)

        l_reshape_2 = lasagne.layers.ReshapeLayer(
            l_LSTM1,
            (l_LSTM1.get_output_shape()[1],
             l_LSTM1.get_output_shape()[2]))

        l_out = lasagne.layers.DenseLayer(
            l_reshape_2,
            num_units=self.number_of_actions,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(std=0.1))

        return l_out

    def train_step(self, input_states, actions, rewards):
        self.states[:len(input_states), :, :, :] = input_states

        qvalues = self.predict(self.states)
        max_qvalues = np.max(qvalues, axis=1)
        max_qvalues = np.roll(max_qvalues, -1)
        max_qvalues[-1] = 0

        qvalues_reinforced = qvalues.copy()

        for i, action in enumerate(actions):
            qvalues_reinforced[i][action] = rewards[i] + \
                self.discount_factor * max_qvalues[i]

        self.train(self.states, qvalues_reinforced)
        self.states.fill(0)

    def greedy_step(self, input_states, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(self.number_of_actions))
        else:
            self.states[:len(input_states), :, :, :] = input_states
            action = np.argmax(self.predict(self.states)[-1])
            self.states.fill(0)

        return action
