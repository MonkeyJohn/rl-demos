import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam


class SimpleLinearQAgent:
	def __init__(self, n_s, n_a, eps, alpha, gamma):
		self.W = np.random.randn(n_s, n_a)
		self.eps = eps
		self.alpha = alpha
		self.gamma = gamma
		self.n_s = n_s
		self.n_a = n_a

		#elf.eps_decay = 0.98
		#self.eps = 1.0

	def Q_sa(self, state, action):
		"state: 1 x n_s, action: int from [0, n_a-1]"
		return np.dot(state.reshape((1,self.n_s)), self.W)[0][action]

	def set_eps(self, eps):
		self.eps = 0

	def get_action(self, state):
		return self.epsilon_greedy(state)

	def epsilon_greedy(self, state):
		if(np.random.random() < self.eps):
			return np.random.choice(range(self.n_a))
		else:
			return self.greedy_action(state)

	def greedy_action(self, state):
		qa = [(self.Q_sa(state, a),a) for a in range(self.n_a)]
		return max(qa, key= lambda x: x[0])[1]

	def greedy_value(self, state):
		qa = [(self.Q_sa(state, a),a) for a in range(self.n_a)]
		return max(qa, key= lambda x: x[0])[0]

	def train(self, state, action, reward, nextState):
		target = reward + self.gamma*self.greedy_value(nextState)
		currQsa = self.Q_sa(state, action)

		grad = state.reshape((self.n_s, 1))
		self.W += self.alpha*(target - currQsa)*grad

		#self.eps *= self.eps_decay
		#print self.eps


def copy_weights(model, target_model):
	for i in range(len(model.layers)):
		target_model.layers[i].set_weights(model.layers[i].get_weights())

class LinearQAgent:
	def __init__(self, n_s, n_a):
		self.n_s = n_s
		self.n_a = n_a

		self.model = Sequential()
		self.model.add(Dense(output_dim=2, input_dim=4, init='uniform', activation='linear'))
		self.model.compile(loss='mse', optimizer='rmsprop')

		self.target_model = Sequential()
		self.target_model.add(Dense(output_dim=2, input_dim=4, init='uniform', activation='linear'))
		copy_weights(self.model, self.target_model)
		self.target_model.compile(loss='mse', optimizer='rmsprop')

		self.eps = 1.0 #exploration factor
		self.eps_decay = 0.005
		self.exploration = 1000

		self.gamma = 0.9 #dicsount factor
		
		self.memory_size = 1000 # number of experiences to train on
		self.experiences = []
		self.batch_size = 200 # minibatch to do gradient descent
		self.target_update_freq = 20
		self.train_freq = 10
		self.steps = 0 # no. of time steps

	def Q_sa(self, state, action):
		"state: 1 x n_s, action: int from [0, n_a-1]"
		return model.predict(state.reshape((1,self.n_s)))[0][action]

	def get_action(self, state):
		return self.epsilon_greedy(state)

	def decay_epsilon(self):
		if(self.eps > 0 and self.steps > self.exploration):
			self.eps -= self.eps_decay

	def set_eps(self, eps):
		self.eps = eps

	def epsilon_greedy(self, state):
		if(np.random.random() < self.eps):
			return np.random.choice(range(self.n_a))
		else:
			return self.greedy_action(state)

	def greedy_action(self, state):
		Q = self.model.predict(state.reshape((1,self.n_s)))[0]
		return np.argmax(Q)

	def record_experience(self, state, action, reward, nextState, episode_end):
		self.experiences.append((state, action, reward, nextState, episode_end))
		if(len(self.experiences) > self.memory_size):
			self.experiences.pop(0)

		self.steps += 1

	def build_train_data(self, size):
		#print("--------WEIGHTS UPDATE----------")
		inp_states = []
		next_states = []
		actions = []
		rewards = []
		episode_ends = []
		mem_size = len(self.experiences)
		exp_arr = np.array(self.experiences)
		sample_exp = exp_arr[np.random.choice(range(mem_size),size,replace=False)]

		for s, a, r, nS, ep_end in sample_exp:
			inp_states.append(s)
			next_states.append(nS)
			actions.append(a)
			rewards.append(r)
			episode_ends.append(ep_end)

		inp_states = np.array(inp_states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)

		Q_next = self.target_model.predict(next_states)
		target = np.zeros_like(Q_next)
		for i in range(size):
			target[i,actions[i]] = rewards[i] + self.gamma*np.max(Q_next[i])*episode_ends[i]

		return inp_states, target

	def train(self):
		if(len(self.experiences) >= self.memory_size):
			if(self.steps % self.train_freq == 0):
				inp_states, target = self.build_train_data(200)
				epochs = 1
				size = self.batch_size
				self.model.fit(inp_states, target, verbose=1, batch_size=20,nb_epoch=epochs)
				self.decay_epsilon()

			if(self.steps % self.target_update_freq == 0):
				copy_weights(self.model, self.target_model)

