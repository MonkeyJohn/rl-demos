import gym
from model import FCNet
from pg_agent import PGAgent

def main():
	env = gym.make('CartPole-v0')
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	model = FCNet(state_dim, 10, action_dim)
	batch_size = 50
	max_episodes = 4000
	agent = PGAgent(env, 'CartPole-v0', model, gamma=0.99, resume=False)

	render=False
	agent.learn(render, 4000, 50, False)




if __name__ == "__main__":
	main()