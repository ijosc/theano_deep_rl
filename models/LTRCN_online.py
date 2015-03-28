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

        l_conv_out = self.build()
        conv_output = l_conv_out.get_output(state)

        LSTM_output = self.LSTM_step(conv_output)

        l_out = lasagne.layers.DenseLayer(
            None,
            num_units=self.number_of_actions,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(std=0.1),
            )
        output = l_out.get_output_for(LSTM_output)

        state = T.tensor4()
        qvalues = T.vector()
        qvalues_reinforced = T.vector()

        self.predict = theano.function([state], output)

        self.sym_hid_prev = T.matrix()
        self.sym_cell_prev = T.matrix()


        objective = lasagne.objectives.Objective(
            self.l_out,
            loss_function=lasagne.objectives.mse)

        cost = objective.get_loss(state, target=qvalues_reinforced)

        all_params = lasagne.layers.get_all_params(self.l_out)
        # add lstm weights and dense layer weights
        # check whether all parameters are in by naming them and/or counting

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

    def LSTM_step(self, input_frame):
        gates = T.dot(input_frame, self.W_in_to_gates) + self.b_gates
        # calculate gates pre-activations and slice
        gates += input_dot_W_n + T.dot(self.hid_previous, self.W_hid_to_gates)
        ingate = slice_w(gates,0)
        forgetgate = slice_w(gates,1)
        modulationgate = slice_w(gates,2)
        outgate = slice_w(gates,3)


        if self.peepholes:
            ingate += self.cell_previous*slice_c(self.W_cell_to_gates, 0)
            forgetgate += self.cell_previous*slice_c(self.W_cell_to_gates, 1)

        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        modulationgate = self.nonlinearity_modulationgate(modulationgate)
        

        cell = forgetgate*self.cell_previous + ingate*modulationgate
        if self.peepholes:
            outgate += cell*slice_c(self.W_cell_to_gates, 2)
        outgate = self.nonlinearity_outgate(outgate)
        hid = outgate*self.nonlinearity_out(cell)
        return [cell, hid]

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
            W=lasagne.init.Normal(std=0.0001),
            )

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            strides=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01),
            )

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            strides=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01),
            )

        l_conv_out = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.1),
            )

        return l_conv_out

    def train_step(self, pre_state, action, reward, post_state):
        qvalues = self.predict(pre_state)[0]
        max_qvalue = np.max(self.predict(post_state))

        qvalues_reinforced = qvalues.copy()
        qvalues_reinforced[action] = reward + \
                                     self.discount_factor * max_qvalue

        self.train( , qvalues_reinforced)

    def greedy_step(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(self.number_of_actions))
        else:
            action = np.argmax(self.predict(state))

        return action

