import gym
from model import FCNet
from pg_agent import PGAgent

def main():
	env = gym.make('CartPole-v0')
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	model = FCNet(state_dim, action_dim)
	batch_size = 50
	max_episodes = 5000
	agent = PGAgent(model, batch_size, max_episodes)

	render= False
	resume= False
	action_space = [0, 1]

	agent.train(env, resume, render, action_space, gamma=0.99)



if __name__ == "__main__":
	main()