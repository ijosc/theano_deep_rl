from utils.ale_interface import ALE
from utils.game_actions import action_dict

from models import DQN
from models import LTRCN
# from models import LTRCN_online

import numpy as np
import sys
import cPickle as pickle

import os


class Main(object):

    frame_size = 84
    state_length = 4
    batch_size = 100

    discount_factor = 0.9
    epsilon_frames = 1000000.0
    total_frames_trained = 0
    q_values = []
    training_scores = []

    test_epsilon = 0.05
    test_scores = []

    ale = None

    def __init__(self, game_name, run_id, model, method):
        self.number_of_actions = len(action_dict[game_name])
        valid_actions = action_dict[game_name]

        self.net = model.Model(
            self.frame_size,
            self.frame_size,
            self.number_of_actions,
            self.batch_size,
            self.state_length)

        self.method = method

        self.ale = ALE(valid_actions,
                       run_id,
                       display_screen="false",
                       skip_frames=4,
                       game_ROM='ale/roms/' + game_name + '.bin')

    def compute_epsilon(self, frames_played):
        return max(1 - frames_played / self.epsilon_frames, 0.1)

    def compute_reward(self, points):
        if points > 0:
            return 1
        elif points < 0:
            return -1
        else:
            return 0

    def play_games(self, nr_frames, epoch, training, epsilon=None):

        frames_played = 0
        game_scores = []

        first_frame = self.ale.new_game()

        if 'current_state' not in locals():
            current_state = np.tile(
                first_frame.copy(),
                (self.state_length, 1)).reshape(1, self.state_length,
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

            if training:
                epsilon = self.compute_epsilon(self.total_frames_trained)

            if training and self.method == 'online':
                self.total_frames_trained += 1

                action = self.net.greedy_step(current_state, epsilon)

                points, next_frame = self.ale.move(action)
                reward = self.compute_reward(points)
                game_score += points

                pre_state = current_state.copy()
                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = next_frame

                self.net.train_step(
                    pre_state,
                    action,
                    reward,
                    current_state)
            else:
                states.append(current_state[0, :, :, :])
                if len(states)>self.batch_size: states.pop(0)
                action = self.net.greedy_step(states, epsilon)

                points, next_frame = self.ale.move(action)
                reward = self.compute_reward(points)
                game_score += points

                actions.append(action)
                rewards.append(reward)
                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = next_frame

            if (training and
                    self.method == 'fixed_frame' and
                    frames_played % self.batch_size == 0):
                self.total_frames_trained += 1
                print 'training with fixed_frame: %d' % self.batch_size
                self.net.train_step(
                    states,
                    actions,
                    rewards)

                actions = []
                rewards = []
                states = []

            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                game_scores.append(game_score)
                game_score = 0

                self.ale.end_game()

                if self.method == 'game_over':
                    self.total_frames_trained += 1
                    self.net.train_step(
                        states,
                        actions,
                        rewards)

                    actions = []
                    rewards = []
                    states = []

                first_frame = self.ale.new_game()
                current_state = np.roll(current_state, -1, 1)
                current_state[0, -1, :, :] = first_frame.copy()

        self.ale.end_game()

        return game_scores

    def run(self, epochs, training_frames, testing_frames):

        for epoch in range(1, epochs + 1):
            print "Epoch %d:" % epoch

            if training_frames > 0:
                print "  Training for %d frames" % training_frames
                self.training_scores.append(
                    self.play_games(training_frames,
                                    epoch,
                                    training=True)
                )

            if testing_frames > 0:
                print "  Testing for %d frames" % testing_frames
                self.test_scores.append(
                    self.play_games(testing_frames,
                                    epoch,
                                    training=False,
                                    epsilon=self.test_epsilon)
                )

        pickle.dump(self.training_scores, open("training_scores.p", "wb"))
        pickle.dump(self.test_scores, open("test_scores.p", "wb"))

if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    training_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    testing_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    run_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    method = 'fixed_frame'

    if os.path.exists('ale_fifo_in_%i' % run_id):
        os.remove('ale_fifo_in_%i' % run_id)
    if os.path.exists('ale_fifo_out_%i' % run_id):
        os.remove('ale_fifo_out_%i' % run_id)

    m = Main('breakout', run_id, LTRCN, method)
    m.run(epochs, training_frames, testing_frames)
