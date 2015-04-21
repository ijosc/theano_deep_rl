
import numpy as np
import sys
import cPickle as pickle

import os

class Main(object):

    def __init__(self, game_name, run_id, model_name):

        self.net = model.Model(
            run_id,
            game_name,
            batch_size=500,
            discount_factor=0.99)

        self.q_values = []
        self.training_scores = []

        self.test_epsilon = 0.05
        self.test_scores = []

    def run(self, training_frames, testing_frames):

        for epoch in range(1, epochs + 1):
            print "Epoch %d:" % epoch

            if training_frames > 0:
                print "  Training for %d frames" % training_frames
                self.training_scores.append(
                    self.play_games(training_frames,
                                    epoch,
                                    training=True))

            if testing_frames > 0:
                print "  Testing for %d frames" % testing_frames
                self.test_scores.append(
                    self.play_games(testing_frames,
                                    epoch,
                                    training=False,
                                    epsilon=self.test_epsilon))

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
