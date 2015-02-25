"""
ALE class launches the ALE game and manages the communication with it
"""

import os
import numpy as np
from preprocessor import Preprocessor
import traceback
import random

class ALE:
    actions = [np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(3), np.uint8(4), np.uint8(5),np.uint8(6), np.uint8(7), np.uint8(8), np.uint8(9), np.uint8(10),np.uint8(11), np.uint8(12),np.uint8(13),np.uint8(14),np.uint8(15),np.uint8(16),np.uint8(17)]

    #space invaders
    legal_actions = [0,1,3,4,11,12]

    current_points = 0
    next_screen = ""
    game_over = False
    skip_frames = None
    display_screen = "true"
    game_ROM = None
    fin = ""
    fout = ""
    preprocessor = None
    lives = 0
    
    def __init__(self, display_screen, skip_frames, game_ROM):
        """
        Initialize ALE class. Creates the FIFO pipes, launches ./ale and does the "handshake" phase of communication

        @param display_screen: bool, whether to show the game on screen or not
        @param skip_frames: int, number of frames to skip in the game emulator
        @param game_ROM: location of the game binary to launch with ./ale
        """

        self.display_screen = display_screen
        self.skip_frames = skip_frames
        self.game_ROM = game_ROM

        #: create FIFO pipes
        os.system("mkfifo ale_fifo_out")
        os.system("mkfifo ale_fifo_in")

        #: launch ALE with appropriate commands in the background
        command='./../libraries/ale/ale -max_num_episodes 0 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip '+str(self.skip_frames)+' -display_screen '+self.display_screen+" "+self.game_ROM+"  -use_environment_distribution &"
        os.system(command)

        #: open communication with pipes
        self.fin = open('ale_fifo_out')
        self.fout = open('ale_fifo_in', 'w')
        
        input = self.fin.readline()[:-1]
        size = input.split("-")  # saves the image sizes (160*210) for breakout

        #: first thing we send to ALE is the output options- we want to get only image data
        # and episode info(hence the zeros)
        self.fout.write("1,1,0,1\n")
        self.fout.flush()  # send the lines written to pipe

        #: initialize the variables that we will start receiving from ./ale
        self.next_image = []
        self.game_over = True
        self.current_points = 0
	self.lives = 0
        #: initialise preprocessor
        self.preprocessor = Preprocessor()

    def new_game(self):
        """
        Start a new game when all lives are lost.
        """

        #: read from ALE:  game screen + episode info
	#delete me
	#test = self.fin.readline()
	#f = open('test.txt', 'w')
	#f.write(str(test))
	#f.close()
        ram, self.next_image, episode_info = self.fin.readline()[:-2].split(":")
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_points = int(episode_info.split(",")[1])
	self.lives = int(ram[201])  #works only for space invaders

        #: send the fist command
        #  first command has to be 1,0 or 1,1, because the game starts when you press "fire!",
        self.fout.write("1,19\n")
        self.fout.flush()
        self.fin.readline()

        #: preprocess the image and add the image to memory D using a special add function
        #self.memory.add_first(self.preprocessor.process(self.next_image))
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
        self.game_over = False  # just in case, but new_game should do it anyway

    
    def move(self, action_index):
        """
        Sends action to ALE and reads responds
        @param action_index: int, the index of the chosen action in the list of available actions
        """
        #: Convert index to action
        action = self.actions[action_index]

	#: Generate a random number for the action of player B
        action_b = np.uint8(18)


        #: Write and send to ALE stuff
        self.fout.write(str(action)+","+str(action_b)+"\n")
        #print "sent action to ALE: ",  str(action)+",",str(action_b)
        self.fout.flush()

        #: Read from ALE
        #line = self.fin.readline()
	#print line
	#raw_input("Press Enter to crash")
        try:
            ram, self.next_image, episode_info = self.fin.readline()[:-2].split(":")
            #print "got correct info from ALE: image + ", episode_info
        except:
            print "got an error in reading stuff from ALE"
            traceback.print_exc()
            print line
            exit()
	#f = open('rams.txt', 'w')
	#f.write(str(ram))
	#f.close()
	#raw_input("Continue...")
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_points = int(episode_info.split(",")[1])
	lives = int(ram[201])  #works only for space invaders
	#print "Lives", self.lives, lives
	if lives < self.lives:
		self.current_points = -1
		print "You Died! Lives left: ", lives
	self.lives = lives
        return self.current_points, self.preprocessor.process(self.next_image)

    def choose_legal_action(self):
        #from the list of available actions, randomly select an index
	#check the self.legal_actions array -> if it contains the index, return the index
	#else start again
        while(1):
            action = random.choice(range(len(self.actions)))
            if action in self.legal_actions:
                return action
