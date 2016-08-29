# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from learningAgents import ReinforcementAgent
import random,util,math

class QLearningAgent(ReinforcementAgent):
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)

		self.QValues = util.Counter()

	def getQValue(self, state, action):
		return self.QValues[(state,action)]

	def getAction(self, state):
		"""
			get action from current state using epsilon-greedy policy
		"""
		if(random.random() < self.epsilon):
			return random.choice(self.getLegalActions(state))
		else:
			return self.greedyAction(state)

	def greedyAction(self, state):
		"""
			Pick the action with maximum QValue
		"""
		legalActions = self.getLegalActions(state)
		if not legalActions : return None

		QValueActions = [(self.getQValue(state, action), action) for action in legalActions]
		maxQValue, maxAction = max(QValueActions, key= lambda x: x[0])

		return random.choice([action for QValue, action in QValueActions if QValue == maxQValue])

	def greedyValue(self, state):
		"""
			Return the maximum Q-Value over all actions
		"""
		legalActions = self.getLegalActions(state)
		if legalActions:
			return max([self.getQValue(state, action) for action in legalActions])
		else:
			return 0


	def update(self, state, action, nextState, reward):
		"""
			Boostrap: Update your QValue for the old_state, action to reward obtained + your estimate of maximum
			QValue of next_state
			Off-Policy Control: Followed policy is epsilon-greedy but update is towards greedy policy
		"""
		sampleQValue = reward + self.discount * self.greedyValue(nextState)
		currQValue = self.getQValue(state, action)
		self.QValues[(state,action)] = currQValue + self.alpha*(sampleQValue - currQValue)