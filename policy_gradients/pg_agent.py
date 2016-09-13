import gym
import numpy as np

def map_action_space(action_space):
	pass
	
class PGAgent:
	def __init__(self, model, batch_size, gamma):
		self.model = model
		self.batch_size = batch_size
		self.gamma = gamma
		self.episode_number = 0

	def train(self, env, env_name, resume, render, action_space, pre_processor=None, save_after=100):
		observation = env.reset()
		prev_x = None
		#xs, hs, dlogps, rs = [],[],[],[]
		running_reward = None
		reward_sum = 0
		self.episode_number = 0

		while True:
			if render: env.render()

			cur_x = pre_processor(observation) if pre_processor not None else observation
			inp_dim = self.model.input_dim
			out_dim = self.model.output_dim
			assert cur_x.shape == inp_dim, "input obs and model input not matching dimension"
			assert len(action_space) == out_dim, "no. of env actions and model out_dim not matching"
			
			x = cur_x - prev_x if prev_x is not None else np.zeros(inp_dim)
			prev_x = cur_x

			action_probs = self.model.forward(x)
			action = np.random.choice(action_space, 1, replace=False, action_probs)

			observation, reward, done, info = env.step(action)
			reward_sum += reward

			temp_act = action - min(action_space)
			self.model.record(x, action, reward)

			if done:
				print "Episode Finished"
				self.episode_number += 1

				self.model.backward() # calculate gradients and keep
				if episode_number % batch_size == 0:
					self.model.update() # update model params

				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print "Resetting env. Episode reward total %f and running mean %f", %(reward_sum, running_reward)

				if self.episode_number % save_after == 0: self.model.save_model("trained_model")
				reward_sum = 0
				observation = env.reset()
				prev_x = None

			# pong specific
			if env_name=='Pong-v0' and reward != 0:
				print ('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')