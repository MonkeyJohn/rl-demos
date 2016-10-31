from environment import AtariAleEnvironment, AtariGymEnvironment
from accum_trainer import AccumTrainer
import tensorflow as tf
from model import ConvNetA3C
import numpy as np
import time

PERF_LOG_INTERVAL = 1000

class A3CTrainer:
	def __init__(self, thread_id, env_name, global_model, init_lr, lr_ph, grad_applier, 
				max_time_steps, model_dim, gamma):
		self.thread_id = thread_id
		self.global_model = global_model
		self.init_lr = init_lr
		self.grad_applier = grad_applier
		self.lr_ph = lr_ph
		self.max_time_steps = max_time_steps
		self.gamma = gamma

		height,width,num_frames,num_actions = model_dim
		self.local_model = ConvNetA3C(height, width, num_frames, num_actions)
		self.num_actions = num_actions
		
		trainer = AccumTrainer("/cpu:0")
		trainer.prepare_minimize(self.local_model.loss, self.local_model.params)
		self.accum_grads = trainer.accumulate_gradients()
		self.reset_grads = trainer.reset_gradients()
		self.apply_grads = grad_applier.apply_gradients(global_model.params, 
												trainer.get_accum_grad_list())

		self.sync = self.local_model.sync_from(global_model)
		self.env = AtariAleEnvironment(env_name)
		self.s_t = self.env.reset()

		self.start_time = None
		self.ep_rwd, self.num_ep = 0, 0
		self.avg_rwd = None
		self.t = 0
		self.prev_t = 0

	def anneal_learning_rate(self, T):
		lr = self.init_lr * (self.max_time_steps-T) / self.max_time_steps
		if lr < 0.0:
			lr = 0.0
		return lr

	def discount_rewards(self, r, gamma, initial=0.0):
		# if r is an integer array, discounted_r also is defined 
		# as integer array and every float value is converted to 
		# integer. This costed me 2 days.
		discounted_r = np.zeros_like(r).astype(np.float32)
		running_add = initial
		for i in reversed(range(0,r.size)):
			running_add = r[i] + running_add*gamma
			discounted_r[i] = running_add

		return discounted_r

	def run_steps(self, sess, t_max, global_t, writer):
		s_n,a_n,r_n,v_n = [],[],[],[]
		
		sess.run(self.reset_grads)
		sess.run(self.sync)

		terminal = False
		start_t = self.t

		for i in range(t_max):
			aprob, value = self.local_model.get_policy_and_value(sess, self.s_t)
			action_index = np.random.choice(self.num_actions, p=aprob)

			s_n.append(self.s_t)
			a_n.append(action_index)
			v_n.append(value)

			#execute action
			self.s_t, r, terminal, _ = self.env.step(action_index)
			self.ep_rwd += r
			r_n.append(np.clip(r,-1,1).astype(np.float32))

			self.t += 1

			if terminal:
				break

		if terminal:
			R = 0.0
		else:
			R = self.local_model.get_value(sess, self.s_t)

		s_n,a_n,r_n,v_n = map(np.array,[s_n,a_n,r_n,v_n])

		disc_r = self.discount_rewards(r_n, self.gamma, R)
		adv = disc_r - v_n
		a_n_one_hot = np.eye(self.num_actions)[a_n]

		sess.run(self.accum_grads, feed_dict={
						self.local_model.states: s_n,
						self.local_model.rewards: disc_r,
						self.local_model.actions: a_n_one_hot,
						self.local_model.advantage: adv
		})


		cur_lr = self.anneal_learning_rate(global_t)

		sess.run(self.apply_grads, feed_dict={
						self.lr_ph: cur_lr
		})

		if terminal:
			self.num_ep += 1
			self.avg_rwd = self.ep_rwd if self.avg_rwd is None\
			 					else 0.99*self.avg_rwd + 0.01*self.ep_rwd

			print "Episode %d, Reward: %d" % (self.num_ep, self.ep_rwd)
			self.global_model.record_summary(sess, writer, self.avg_rwd, 
												self.ep_rwd, global_t)

			if self.thread_id == 0:
				print "Probs: ", aprob, "Value: ", value, '\n'

			self.ep_rwd = 0
			self.s_t = self.env.reset()

		if self.thread_id == 0 and self.t - self.prev_t >= PERF_LOG_INTERVAL:
			self.prev_t += PERF_LOG_INTERVAL
			elapsed_time = time.time() - self.start_time
			steps_sec = global_t / elapsed_time
			print "Performance: Steps %d in %d sec, %d steps per sec, %.2fM steps an hour\n"\
							%(global_t, elapsed_time, steps_sec, steps_sec*(3600/1e6))

		return self.t - start_t


