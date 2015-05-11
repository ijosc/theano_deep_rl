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

from utils.ale_interface import ALE
from utils.game_actions import action_dict


class Model(object):

    def __init__(self, run_id, game_name, learning_rate,
                 batch_size=100, discount_factor=0.99):

        self.frame_size = 84
        self.channels = 4
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.discount_factor = discount_factor
        self.epsilon_frames = 1000000.0
        self.total_frames_trained = 0
        self.test_epsilon = 0.05

        self.states = np.zeros((self.batch_size,
                                self.channels,
                                self.frame_size,
                                self.frame_size))

        self.number_of_actions = len(action_dict[game_name])
        valid_actions = action_dict[game_name]

        self.ale = ALE(valid_actions,
                       run_id,
                       display_screen="false",
                       skip_frames=4,
                       game_ROM='ale/roms/' + game_name + '.bin')

        self.net = self.build_net()

        self.setup_theano()

    def setup_theano(self):

        state = T.tensor4()
        qvalues = T.vector()
        qvalues_reinforced = T.matrix()

        output = self.net.get_output(state)

        self.predict = theano.function(
            [state],
            output,
            allow_input_downcast=True)

        objective = lasagne.objectives.Objective(
            self.net,
            loss_function=lasagne.objectives.mse)

        cost = objective.get_loss(state, target=qvalues_reinforced)

        all_params = lasagne.layers.get_all_params(self.net)

        updates = lasagne.updates.rmsprop(
            cost,
            all_params,
            learning_rate=self.learning_rate,
            rho=0.9,
            epsilon=1e-06)

        self.train_net = theano.function(
            inputs=[state, qvalues_reinforced],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)

    def build_net(self):
        l_in = lasagne.layers.InputLayer(
            shape=(
                self.batch_size,
                self.channels,
                self.frame_size,
                self.frame_size))

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

        l_LSTM_1 = lasagne.layers.LSTMLayer(
            l_reshape_1,
            num_units=102,
            peepholes=True,
            learn_init=True)

        l_reshape_2 = lasagne.layers.ReshapeLayer(
            l_LSTM_1,
            (l_LSTM_1.get_output_shape()[1],
             l_LSTM_1.get_output_shape()[2]))

        l_out = lasagne.layers.DenseLayer(
            l_reshape_2,
            num_units=self.number_of_actions,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(std=0.1))

        return l_out

    def train_step(self, input_states, actions, rewards):
        self.states[:len(input_states), :, :, :] = input_states

        qvalues = self.predict(self.states)

        
        max_qvalues_s = np.max(qvalues, axis=1)
        # we want to find the max q value for s':

        max_qvalues_sPrime = np.roll(max_qvalues_s, -1)
        max_qvalues_sPrime[-1] = 0

        qvalues_reinforced = qvalues.copy()

        for i, action in enumerate(actions):
            qvalues_reinforced[i][action] = rewards[i] + \
                self.discount_factor * max_qvalues_sPrime[i]

        self.train_net(self.states, qvalues_reinforced)
        self.states.fill(0)

        return sum(max_qvalues_s)/len(max_qvalues_s)

    def eps_greedy_step(self, input_states, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(self.number_of_actions))
        else:
            self.states[:len(input_states), :, :, :] = input_states
            action = np.argmax(self.predict(self.states)[-1])
            self.states.fill(0)

        return action

    def train(self, nr_frames):

        frames_played = 0
        game_scores = []
        max_qvalues = []

        first_frame = self.ale.new_game()

        if 'current_state' not in locals():
            current_state = np.tile(
                first_frame.copy(),
                (self.channels, 1)).reshape(1, self.channels,
                                                self.frame_size,
                                                self.frame_size)
        else:
            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = first_frame.copy()

        game_score = 0
        actions = []
        rewards = []
        states = []

        while frames_played < nr_frames:
            frames_played += 1
            self.total_frames_trained += 1

            epsilon = self.compute_epsilon(self.total_frames_trained)

            states.append(current_state[0, :, :, :])
            if len(states)>self.batch_size: states.pop(0)
            action = self.eps_greedy_step(states, epsilon)

            points, next_frame = self.ale.move(action)
            reward = self.compute_reward(points)
            game_score += points

            actions.append(action)
            rewards.append(reward)
            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = next_frame

            if self.ale.game_over:
                game_score = 0

                self.ale.end_game()

                max_qvalues.append(self.train_step(
                    states,
                    actions,
                    rewards))

                actions = []
                rewards = []
                states = []

                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = self.ale.new_game()

        self.ale.end_game()

        return game_scores, max_qvalues


    def test(self, nr_frames):
        frames_played = 0
        game_scores = []

        first_frame = self.ale.new_game()

        if 'current_state' not in locals():
            current_state = np.tile(
                first_frame.copy(),
                (self.channels, 1)).reshape(1, self.channels,
                                                self.frame_size,
                                                self.frame_size)
        else:
            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = first_frame.copy()

        game_score = 0
        actions = []
        rewards = []
        states = []

        while frames_played < nr_frames:
            frames_played += 1

            states.append(current_state[0, :, :, :])
            if len(states)>self.batch_size: states.pop(0)
            action = self.eps_greedy_step(states, self.test_epsilon)

            points, next_frame = self.ale.move(action)
            reward = self.compute_reward(points)
            game_score += points

            actions.append(action)
            rewards.append(reward)
            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = next_frame

            if self.ale.game_over:
                game_scores.append(game_score)
                game_score = 0

                self.ale.end_game()

                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = self.ale.new_game()

        self.ale.end_game()

        return game_scores

    def compute_epsilon(self, frames_played):
        return max(1 - frames_played / self.epsilon_frames, 0.1)

    def compute_reward(self, points):
        if points > 0:
            return 1
        elif points < 0:
            return -1
        else:
            return 0
