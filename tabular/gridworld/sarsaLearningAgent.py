from learningAgents import ReinforcementAgent
import random, util

class SarsaLearningAgent(ReinforcementAgent):
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)

		self.QValues = util.Counter()

		self.sampleAction = None # For SARSA specific update

	def getQValue(self, state, action):
		return self.QValues[(state,action)]

	def epsilon_greedy(self, state):
		"""
			Returns a random action with probability epsilon and greedy action with 1-epsilon
		"""
		if(random.random() < self.epsilon):
			return random.choice(self.getLegalActions(state))
		else:
			return self.greedyAction(state)

	def getAction(self, state):
		"""
			get action from current state using epsilon-greedy policy
		"""
		if self.sampleAction:
			return self.sampleAction
		else:
			return self.epsilon_greedy(state)


	def greedyAction(self, state):
		"""
			Pick the action with maximum QValue
		"""
		legalActions = self.getLegalActions(state)
		if not legalActions : return None

		QValueActions = [(self.getQValue(state, action), action) for action in legalActions]
		maxQValue, maxAction = max(QValueActions, key= lambda x: x[0])

		return random.choice([action for QValue, action in QValueActions if QValue == maxQValue])

	def sampleQValue(self, state):
		"""
			For SARSA, after we know nextState, we sample an action from the current policy, and return
			the corresponding QValue.
		"""
		if not self.getLegalActions(state): return 0
		
		self.sampleAction = self.epsilon_greedy(state)
		return self.getQValue(state, self.sampleAction)

	def update(self, state, action, nextState, reward):
		"""
			Sarsa Bootstrap: Update QValue for oldState,action pair to new reward + estimate of QValue for
			nextState, sample action using current policy.
			This sample action will actually be taken in the next step
			This is an On-Policy control because update happens towards QValue of an action sampled from current
			policy. 
		"""
		sample = reward + self.discount * self.sampleQValue(nextState)
		currQValue = self.getQValue(state, action)
		self.QValues[(state,action)] = currQValue + self.alpha*(sample - currQValue)