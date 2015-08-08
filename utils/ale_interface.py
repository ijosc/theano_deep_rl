import os
import numpy as np
from preprocessor import Preprocessor
import traceback
import random
import sys

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)


class ALE:
    all_actions = []
    for i in range(18):
        all_actions.append(np.uint8(i))
    current_points = 0
    next_screen = ""
    game_over = False
    skip_frames = None
    display_screen = "true"
    game_ROM = None
    fin = ""
    fout = ""
    preprocessor = None

    def __init__(self, valid_actions, run_id, display_screen, skip_frames, game_ROM):
        self.display_screen = display_screen
        self.skip_frames = skip_frames
        self.game_ROM = game_ROM
        self.run_id = run_id

        #: create FIFO pipes
        os.mkfifo("ale_fifo_out_%i" % self.run_id)
        os.mkfifo("ale_fifo_in_%i" % self.run_id)

        #: launch ALE with appropriate commands in the background
        command = './ale/ale -max_num_episodes 0 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip ' + \
            str(self.skip_frames) + ' -run_id ' + str(self.run_id) + \
            ' -display_screen ' + self.display_screen + \
            " " + self.game_ROM + " &"
        os.system(command)

        os.system('ls -l ale_fifo_out_%i' % self.run_id)
        os.system('ls -l ale_fifo_in_%i' % self.run_id)

        #: open communication with pipes

        self.fin = open('ale_fifo_out_%i' % self.run_id)
        self.fout = open('ale_fifo_in_%i' % self.run_id, 'w')

        input = self.fin.readline()[:-1]
        size = input.split("-")  # saves the image sizes (160*210) for breakout

        #: first thing we send to ALE is the output options- we want to get only image data
        # and episode info(hence the zeros)
        self.fout.write("1,0,0,1\n")
        self.fout.flush()  # send the lines written to pipe

        #: initialize the variables that we will start receiving from ./ale
        self.next_image = []
        self.game_over = True
        self.current_points = 0
        self.actions = [self.all_actions[i] for i in valid_actions]

        #: initialise preprocessor
        self.preprocessor = Preprocessor()

    def new_game(self):

        #: read from ALE:  game screen + episode info
        self.next_image, episode_info = self.fin.readline()[:-2].split(":")
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_points = int(episode_info.split(",")[1])

        #: send the fist command
        # first command has to be 1,0 or 1,1, because the game starts when you
        # press "fire!",
        self.fout.write("1, 19\n")
        self.fout.flush()
        self.fin.readline()

        return self.preprocessor.process(self.next_image)

    def end_game(self):
        """
        When all lives are lost, end_game adds last frame to memory resets the system
        """
        #: tell the memory that we lost
        # self.memory.add_last() # this will be done in Main.py

        #: send reset command to ALE
        self.fout.write("45,45\n")
        self.fout.flush()
        # just in case, but new_game should do it anyway
        self.game_over = False

    def move(self, action_index):
        """
        Sends action to ALE and reads responds
        @param action_index: int, the index of the chosen action in the list of available actions
        """
        #: Convert index to action
        action = self.actions[action_index]

        #: Write and send to ALE stuff

        self.fout.write(str(action) + "," + str(19) + "\n")
        # self.fout.write(str(action)+"\n")
        self.fout.flush()
        #: Read from ALE
        line = self.fin.readline()
        try:
            self.next_image, episode_info = line[:-2].split(":")
            # print "got correct info from ALE: image + ", episode_info
        except:
            print "got an error in reading stuff from ALE"
            traceback.print_exc()
            print line
            exit()
        self.game_over = bool(int(episode_info.split(",")[0]))

        self.current_points = int(episode_info.split(",")[1])
        return self.current_points, self.preprocessor.process(self.next_image)
