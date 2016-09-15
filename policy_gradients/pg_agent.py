import numpy as np

	
class PGAgent:
	def __init__(self, model, batch_size, max_episodes):
		self.model = model
		self.batch_size = batch_size
		self.episode_number = 0
		self.max_episodes = max_episodes

	def train(self, env, resume, render, action_space, pre_processor=None, save_after=100, map_action=None, **kwargs):
		observation = env.reset()
		prev_x = None
		#xs, hs, dlogps, rs = [],[],[],[]
		running_reward = None
		reward_sum = 0
		self.episode_number = 0
		self.model.initialize()

		while self.episode_number <= self.max_episodes:
			if render: env.render()

			inp_dim = self.model.inp_dim
			out_dim = self.model.out_dim
			cur_x = pre_processor(observation) if pre_processor is not None else observation.reshape((1, inp_dim))
			assert cur_x.shape[1] == inp_dim, "input obs and model input not matching dimension"
			#assert len(action_space) == out_dim, "no. of env actions and model out_dim not matching"
			
			#x = cur_x - prev_x if prev_x is not None else np.zeros((1,inp_dim))
			#prev_x = cur_x
			x = cur_x

			action_probs = self.model.predict(x)
			#print action_probs
			action = np.random.choice(action_space, 1, replace=False, p=action_probs[0])[0]
			#tfprob = self.model.predict(x)
			#action = 1 if np.random.uniform() < tfprob else 0
			observation, reward, done, info = env.step(action)
			reward_sum += reward

			if map_action:
				action = map_action(action_space, action)

			self.model.record(x, action, reward)

			if done:
				#print "Episode Finished"
				self.episode_number += 1

				self.model.accumulate_gradients(**kwargs) # calculate gradients and keep
				if self.episode_number % self.batch_size == 0:
					print 'Updating Params!!'
					self.model.update_params() # update model params
					print 'Ep No %d, Average Reward per episode %f' % (self.episode_number, reward_sum/self.batch_size)
					reward_sum = 0

				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				#print "Episode number %d, Episode reward total %f and running mean %f" %(self.episode_number, reward_sum, running_reward)

				self.model.record_summary(reward_sum, self.episode_number)
				#if self.episode_number % save_after == 0: self.model.save_model(self.episode_number)
				#reward_sum = 0
				observation = env.reset()
				prev_x = None

			# pong specific
			if 'env_name' in kwargs:
				if kwargs['env_name']=='Pong-v0' and reward != 0:
					print ('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')