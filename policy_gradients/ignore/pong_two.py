import gym
from model import FCNet
from pg_agent import PGAgent
import numpy as np

def map_action(action_space, action, env_to_agent=True):
	if env_to_agent:
		d = {2: 0, 3: 1}
		return d[action]
	else:
		d = {0: 2, 1: 3}
		return d[action]

def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel().reshape((1,6400))

def main():
	env = gym.make('Pong-v0')
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	model = FCNet(6400, 200, 2)
	agent = PGAgent(env, 'Pong-v0', model, gamma=0.99, pre_processor=prepro, action_space=[2,3], map_action=map_action)

	render=False
	agent.learn(render, 10000, 10, diff_frame=True)




if __name__ == "__main__":
	main()