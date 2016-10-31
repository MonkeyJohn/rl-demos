## adapted from coreylynch/async-rl

from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque

class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state
    is constructed.
    """
    def __init__(self, gym_env, height, width, agent_history_length):
        self.env = gym_env
        self.height = height
        self.width = width
        self.agent_history_length = agent_history_length

        self.gym_actions = range(gym_env.action_space.n)
        if (gym_env.spec.id == "Pong-v0" or gym_env.spec.id == "Breakout-v0"):
            print "Doing workaround for pong or breakout"
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

        # Screen buffer of size AGENT_HISTORY_LENGTH-1 to be able
        # to build state arrays of size [1, height, width, AGENT_HISTORY_LENGTH]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
        
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (self.height, self.width))

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        nf = self.agent_history_length
        previous_frames = np.array(self.state_buffer).transpose((1,2,0))
        s_t1 = np.empty((self.height, self.width, nf))
        s_t1[:,:,:nf-1] = previous_frames
        s_t1[:,:,nf-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info
