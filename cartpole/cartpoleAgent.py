import gym
import sys
from qlearn import LinearQAgent, SimpleLinearQAgent
import numpy as np
import time

def run(agent, env, num_eps, max_iter, render=False, monitor=False, train=True):
	if monitor:
		env.monitor.start('log/exp1', force=True)

	if not train:
		agent.set_eps(0)
	total = 0
	for n in range(num_eps):
		reward_total = 0
		observation = env.reset()

		for i in range(max_iter):
			if render:
				env.render()

			state = observation.astype(np.float32)
			action = agent.get_action(state)
			observation, reward, done, _ = env.step(action)
			nextState = observation.astype(np.float32)
			reward_total += reward

			if train:
				agent.record_experience(state,action,reward,nextState,done)
				agent.train()
				#agent.train(state,action,reward,nextState)

			if done:
				break

		print "Episode %d, Total Reward %d" %(n+1, reward_total)

		total += reward_total

	if monitor:
		env.monitor.close()

	print "Average Reward: ", float(total)/num_eps

def main(num_eps, max_iter, render=False, monitor=True):
	env = gym.make('CartPole-v0')
	n_s = env.observation_space.shape[0]
	n_a = env.action_space.n

	agent = LinearQAgent(n_s, n_a)
	#agent = SimpleLinearQAgent(n_s, n_a, 0.3, 0.2, 0.9)
	run(agent, env, num_eps, max_iter, render, monitor, train=True)

	print "-------------DONE TRAINING-------------------"
	run(agent, env, num_eps, max_iter, render, monitor, train=False)

if __name__ == "__main__":
	main(500, 1000, render=True, monitor=False)
	sys.exit(0)





