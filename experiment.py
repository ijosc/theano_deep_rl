from sacred import Experiment
from sacred.observers import MongoObserver
from models import *

import os
import sys

ex = Experiment('LTRCN')

db_user = 'sacred'
db_password = 'sacred119'

ex.observers.append(MongoObserver.create(
    url='mongodb://'+db_user+':'+db_password+'@ds031932.mongolab.com:31932/mt_experiments',
    db_name='mt_experiments'))

@ex.config
def my_config():
    game_name = 'breakout'
    model_name = 'DQN'

    n_epochs = 10
    training_frames = 50000
    testing_frames = 1000

    run_id = 1

@ex.automain
def ex_main(game_name, model_name,
            n_epochs, training_frames, testing_frames,
            run_id):

    if os.path.exists('ale_fifo_in_%i' % run_id):
        os.remove('ale_fifo_in_%i' % run_id)

    if os.path.exists('ale_fifo_out_%i' % run_id):
        os.remove('ale_fifo_out_%i' % run_id) 

    if model_name == 'LTRCN':
        model = LTRCN.Model(
            run_id,
            game_name,
            batch_size=512,
            discount_factor=0.99)
    elif model_name == 'DQN':
        model = DQN.Model(
            run_id,
            game_name,
            batch_size=32,
            discount_factor=0.99)


    for epoch in range(n_epochs):
        print "Epoch %d:" % (epoch+1)

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

        if testing_frames > 0:
            test_scores = model.test(testing_frames)

            if 'test_scores' in ex.info and len(test_scores) is not 0:
                ex.info['test_scores'].append(float(sum(test_scores))/len(test_scores))
            elif 'test_scores' not in ex.info and len(test_scores) is not 0:
                ex.info['test_scores'] = [float(sum(test_scores))/len(test_scores)]