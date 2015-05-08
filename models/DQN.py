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
from utils.memory import Memory


class Model(object):

    def __init__(self, run_id, game_name,
                 batch_size=32, discount_factor=0.99):

        self.frame_size = 84
        self.channels = 4
        self.batch_size = batch_size

        memory_size = 500000
        self.memory = Memory(memory_size)

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
            learning_rate=0.00001,
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

        l_out = lasagne.layers.DenseLayer(
            l_dense,
            num_units=self.number_of_actions,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(std=0.1))

        return l_out

    def train_step(self, minibatch):

        prestates, actions, rewards, poststates = minibatch

        qvalues = self.predict(prestates)
        post_qvalues = self.predict(poststates)
        
        max_qvalues = np.max(post_qvalues,axis=1)

        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + \
                self.discount_factor * max_qvalues[i]

        self.train_net(prestates, qvalues)

        return sum(max_qvalues)/len(max_qvalues)

    def eps_greedy_step(self, states, epsilon):
        if random.uniform(0, 1) < epsilon:
            return (random.choice(range(self.number_of_actions)))
        else:
            return (np.argmax(self.predict(states)[0]))

    def train(self, nr_frames):

        frames_played = 0
        game_scores = []
        max_qvalues = []

        first_frame = self.ale.new_game()
        self.memory.add_first(first_frame)

        states = np.zeros(( self.batch_size,
                            self.channels,
                            self.frame_size,
                            self.frame_size))

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


        while frames_played < nr_frames:

            frames_played += 1
            self.total_frames_trained += 1
            epsilon = self.compute_epsilon(self.total_frames_trained)

            states[0, :, :, :] = current_state
            action = self.eps_greedy_step(states, epsilon)

            points, next_frame = self.ale.move(action)
            reward = self.compute_reward(points)
            game_score += points

            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = next_frame

            self.memory.add(action, reward, next_frame)

            minibatch = self.memory.get_minibatch(self.batch_size)
            max_qvalues.append(self.train_step(minibatch))

            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                game_scores.append(game_score)
                game_score = 0

                self.ale.end_game()
                self.memory.add_last()

                first_frame = self.ale.new_game()
                self.memory.add_first(first_frame)

                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = first_frame.copy()

        self.ale.end_game()

        return game_scores, max_qvalues


    def test(self, nr_frames):
        frames_played = 0
        game_scores = []
        max_qvalues = []

        first_frame = self.ale.new_game()

        states = np.zeros(( self.batch_size,
                            self.channels,
                            self.frame_size,
                            self.frame_size))

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


        while frames_played < nr_frames:
            frames_played += 1

            epsilon = self.test_epsilon

            states[0, :, :, :] = current_state
            action = self.eps_greedy_step(states, epsilon)

            points, next_frame = self.ale.move(action)
            reward = self.compute_reward(points)
            game_score += points

            current_state = np.roll(current_state, -1, 1)
            current_state[0, -1, :, :] = next_frame

            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                game_scores.append(game_score)
                game_score = 0

                self.ale.end_game()

                first_frame = self.ale.new_game()

                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = first_frame.copy()

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
