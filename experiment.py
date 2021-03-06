from sacred import Experiment
from sacred.observers import MongoObserver
from models import *

import os
import sys
import cPickle as pickle

ex = Experiment('LTRCN')

@ex.config
def my_config():
    game_name = 'ms_pacman'
    model_name = 'LTRCN'
    model_path = 'LTRCN_ms_pacman_3.p'
    total_frames_trained = 0
    batch_size = 120
    learning_rate = 0.001

    n_epochs = 2
    training_frames = 0
    testing_frames = 200

    run_id = 99

@ex.automain
def ex_main(game_name, model_name, model_path, batch_size, learning_rate,
            n_epochs, training_frames, testing_frames, total_frames_trained,
            run_id):

    if os.path.exists('ale_fifo_in_%i' % run_id):
        os.remove('ale_fifo_in_%i' % run_id)

    if os.path.exists('ale_fifo_out_%i' % run_id):
        os.remove('ale_fifo_out_%i' % run_id)

    if model_name == 'LTRCN':
        model = LTRCN.Model(
            run_id,
            game_name,
            learning_rate,
            batch_size=batch_size,
            discount_factor=0.99)
    elif model_name == 'DQN':
        model = DQN.Model(
            run_id,
            game_name,
            learning_rate,
            batch_size=batch_size,
            discount_factor=0.99)
    elif model_name == 'LTRCN_gameover':
        model = LTRCN_gameover.Model(
            run_id,
            game_name,
            learning_rate,
            batch_size=batch_size,
            discount_factor=0.99)
    else:
        print 'Model name is not recognized'

    if model_path != '':
        model.load_params(model_path)
        model.total_frames_trained = total_frames_trained

    for epoch in range(n_epochs):
        print 'Epoch %d:' % (epoch+1)

        if training_frames > 0:
            training_scores, max_qvalues = model.train(training_frames)

            if 'max_qvalues' in ex.info:
                ex.info['max_qvalues'].append(sum(max_qvalues)/len(max_qvalues))
            else:
                ex.info['max_qvalues'] = [sum(max_qvalues)/len(max_qvalues)]

            if 'training_scores' in ex.info and len(training_scores) is not 0:
                ex.info['training_scores'].append(float(sum(training_scores))/len(training_scores))
            elif 'training_scores' not in ex.info and len(training_scores) is not 0:
                ex.info['training_scores'] = [float(sum(training_scores))/len(training_scores)]

            model.save_params('%s_%s_%d.p'%(model_name, game_name, run_id))

        if testing_frames > 0:
            test_scores = model.test(testing_frames)

            if 'test_scores' in ex.info and len(test_scores) is not 0:
                ex.info['test_scores'].append(float(sum(test_scores))/len(test_scores))
            elif 'test_scores' not in ex.info and len(test_scores) is not 0:
                ex.info['test_scores'] = [float(sum(test_scores))/len(test_scores)]
        print model.cost
        print model.total_frames_trained
        ex.info['Cost'] = model.cost
        ex.info['Total Frames Trained'] = model.total_frames_trained
