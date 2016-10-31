import cv2
import numpy as np
from collections import deque
import gym
import sys
from ale_python_interface import ALEInterface
from skimage.color import rgb2gray

## adapted from coreylynch/async-rl
class AtariGymEnvironment(object):
	"""
	Small wrapper for gym atari environments.
	Responsible for preprocessing screens and holding on to a screen buffer 
	of size agent_history_length from which environment state
	is constructed.
	"""
	def __init__(self, env_name, height, width, num_frames):
		self.env = gym.make(env_name)
		self.height = height
		self.width = width
		self.num_frames = num_frames

		self.gym_actions = range(self.env.action_space.n)
		if (self.env.spec.id == "Pong-v0" or self.env.spec.id == "Breakout-v0"):
			print "Doing workaround for pong or breakout"
			# Gym returns 6 possible actions for breakout and pong.
			# Only three are used, the rest are no-ops. This just lets us
			# pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
			self.gym_actions = [1,2,3]


	def reset(self):
		"""
		Resets the atari game
		"""
		# Clear the state buffer

		x_t = self.env.reset()
		x_t = self._get_preprocessed_frame(x_t)
		self.s_t = np.stack([x_t for i in range(self.num_frames)], axis = 2)
		
		return self.s_t

	def _get_preprocessed_frame(self, observation):
		"""
		See Methods->Preprocessing in Mnih et al.
		1) Get image grayscale
		2) Rescale image
		3) Returns image of shape (height, width)
		"""

		return cv2.resize(rgb2gray(observation),(self.width, self.height))

	def step(self, action_index):
		"""
		Excecutes an action in the gym environment.
		Builds current state (concatenation of agent_history_length-1 previous frames and current one).
		Pops oldest frame, adds current frame to the state buffer.
		Returns current state.
		"""

		x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
		x_t1 = self._get_preprocessed_frame(x_t1)

		h,w = x_t1.shape
		x_t1 = x_t1.reshape(h,w,1)
		s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)
		self.s_t = s_t1

		return s_t1, r_t, terminal, info



# adapted from miyosuda/async_deep_reinforce
class AtariAleEnvironment(object):
	def __init__(self, env_name,display=False, no_op_max=7):
		self.ale = ALEInterface()
		self.ale.setInt(b'random_seed',113*np.random.randint(0,5))
		self.ale.setFloat(b'repeat_action_probability', 0.0)
		self.ale.setBool(b'color_averaging', True)
		self.ale.setInt(b'frame_skip', 4)
		self._no_op_max = no_op_max

		if display:
			self._setup_display()

		rom_name = env_name + '.bin'
		self.ale.loadROM(rom_name.encode('ascii'))

		# collect minimal action set
		self.real_actions = self.ale.getMinimalActionSet()

		# height=210, width=160
		self._screen = np.empty((210,160, 1), dtype=np.uint8)


	def _process_frame(self, action, reshape):
		reward = self.ale.act(action)
		terminal = self.ale.game_over()

		# screen shape is (210, 160, 1)
		self.ale.getScreenGrayscale(self._screen)

		# reshape it into (210, 160)
		reshaped_screen = np.reshape(self._screen, (210, 160))

		# resize to height=110, width=84
		resized_screen = cv2.resize(reshaped_screen, (84, 84))

		#x_t = resized_screen[18:102,:]
		x_t = resized_screen
		if reshape:
			x_t = np.reshape(x_t, (84, 84, 1))
		x_t = x_t.astype(np.float32)
		x_t *= (1.0/255.0)
		return reward, terminal, x_t
	
	
	def _setup_display(self):
		if sys.platform == 'darwin':
			import pygame
			pygame.init()
			self.ale.setBool(b'sound', False)
		elif sys.platform.startswith('linux'):
			self.ale.setBool(b'sound', True)
		self.ale.setBool(b'display_screen', True)

	def reset(self):
		self.ale.reset_game()

		# randomize initial state
		if self._no_op_max > 0:
			no_op = np.random.randint(0, self._no_op_max + 1)
			for _ in range(no_op):
				self.ale.act(0)

		_, _, x_t = self._process_frame(0, False)

		self.reward = 0
		self.terminal = False
		self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

		return self.s_t
	
	def step(self, action):
		# convert original 18 action index to minimal action set index
		real_action = self.real_actions[action]

		r, t, x_t1 = self._process_frame(real_action, True)

		self.reward = r
		self.terminal = t
		s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)   
		self.s_t = s_t1

		# 4th argument is some info from gym; consistency hack
		return self.s_t, self.reward, self.terminal, None