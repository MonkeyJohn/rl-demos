import numpy as np

	
class PGAgent:
	def __init__(self, env, env_name, model, resume=False, gamma=0.99, pre_processor=None, action_space=None, map_action=None):
		self.env = env
		self.env_name = env_name
		self.model = model
		self.gamma = gamma
		self.pre_processor = pre_processor

		s_shape = env.observation_space.shape
		a_dim = env.action_space.n
		if pre_processor:
			self.inp_shape = pre_processor(np.zeros(s_dim)).shape
		else:
			self.inp_shape = s_shape

		self.inp_dim = np.product(self.inp_shape)
		self.out_dim = a_dim if action_space is None else len(action_space)

		assert self.inp_dim == self.model.inp_dim, "Agent state dim and Model input dim not matching!"
		assert self.out_dim == self.model.out_dim, "Agent action dim and Model output dim not matching!"

		self.map_action = map_action
		self.action_space = range(a_dim) if action_space is None else action_space

		self.train = True
		if resume:
			self.model.load_model()
			self.train = False


	def learn(self, render, max_episodes, batch_size, diff_frame=True, save_after=1000):
		env, env_name = self.env, self.env_name
		observation = env.reset()
		running_reward = None
		reward_sum = 0
		episode_number = 0
		self.model.initialize()

		prev_x = None
		while episode_number <= max_episodes:
			if render: env.render()
			
			cur_x = self.pre_processor(observation) if self.pre_processor else observation.reshape((1,self.inp_dim))

			if diff_frame:
				x = cur_x - prev_x if prev_x is not None else np.zeros((1,self.inp_dim))
				prev_x = cur_x
			else:
				x = cur_x

			action_probs = self.model.predict(x).flatten()
			action = np.random.choice(self.action_space, 1, replace=False, p=action_probs)[0]
			observation, reward, done, info = env.step(action)
			reward_sum += reward

			if self.map_action:
				action = self.map_action(self.action_space, action)

			self.model.record(x, action, reward)

			if done:
				episode_number += 1
				discounted_rewards = self.discount_rewards(np.vstack(self.model.rs))

				if self.train:
					self.model.calculate_gradients(discounted_rewards) # calculate gradients and keep

				if episode_number % batch_size == 0:
					if self.train:
						print 'Updating Params!!'
						self.model.update_params() # update model params
					print 'Ep No %d, Average Reward per episode %f' % (episode_number, reward_sum/float(batch_size))
					reward_sum = 0

				if self.train and episode_number % save_after == 0:
					print "Saving Model!"
					self.model.save_model()

				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				self.model.record_summary(reward_sum, episode_number)
				observation = env.reset()
				prev_x = None

			# pong specific
			if self.env_name=='Pong-v0' and reward != 0:
				print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')



	def discount_rewards(self, r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		gamma = self.gamma
		for t in reversed(xrange(0, r.size)):
			if self.env_name == 'Pong-v0': # Pong Specific
				if(r[t] != 0): runnnig_add = 0
			running_add = running_add * gamma + r[t]
			discounted_r[t] = running_add
		
		discounted_r -= np.mean(discounted_r)	
		discounted_r /= np.std(discounted_r)
		return discounted_r