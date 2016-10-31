import tensorflow as tf
from util_pg import discount_rewards
import gym
import numpy as np
from model import TwoLayerNetA3C
import time
import threading
from keras import backend as K

tf.app.flags.DEFINE_string("env", "CartPole-v0", "Gym environment to run")
tf.app.flags.DEFINE_float("gamma", .99, "Discount factor")
tf.app.flags.DEFINE_integer("num_episodes", 2000,
                            "Total no. of episodes to run")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Maximum number of updates")
tf.app.flags.DEFINE_integer("num_actions", 2, "Number of actions for agent")
tf.app.flags.DEFINE_boolean("train", True, "Training phase or Eval phase")

FLAGS = tf.app.flags.FLAGS

MAX_TIME_STEPS = 3*10**5
NUM_ACTIONS = 3
NUM_THREADS = 2
LOGDIR_TRAIN = FLAGS.env.strip('-v0') + "-a3c_train_logs/"
LOGDIR_EVAL  = FLAGS.env.strip('-v0') + "-a3c_test_logs/"
SAVE_LOG_INTERVAL = 10000
PERFORMANCE_LOG_INTERVAL = 500
T = 0


def log_uniform(lo, hi, rate):
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return np.exp(v)

LR_LOW = 1e-3
LR_HIGH = 1e-2
RATE = 0.4226
INIT_LR = log_uniform(LR_LOW, LR_HIGH, RATE)


def anneal_learning_rate(T):
    lr = INIT_LR * (MAX_TIME_STEPS - T) / MAX_TIME_STEPS
    if lr < 0.0:
      lr = 0.0
    return lr

def initialize_uninit_var(sess):
    unit_vars = [v for v in tf.all_variables() if not\
                    sess.run(tf.is_variable_initialized(v))]
    sess.run(tf.initialize_variables(unit_vars))

def a3c_thread(t_id, env, sess, global_model, writer, saver):
    global T
    #K.set_session(sess)
    time.sleep(2*(t_id+1))
    env.seed(7)
    #env = AtariEnvironment(env, HEIGHT, WIDTH, AGENT_HISTORY_LENGTH)
    actions = [0,1]
    with sess.graph.as_default():
        local_model = TwoLayerNetA3C(4,2)
        initialize_uninit_var(sess)
        sync = local_model.sync_from(global_model)
    # do this. you have some variables OUTSIDE Keras which Keras can't initialize
    
    gamma, env_name = FLAGS.gamma, FLAGS.env
    
    t_max = 32
    avg_reward  = None
    n_e = 0
    terminal = False
    ep_rwd = 0
    t = 0
    prev_t, prev_t_save = 0, 0
    start_time = time.time()

    observation = env.reset()
    while T < MAX_TIME_STEPS:
        sess.run(sync)
        o_n, a_n, r_n, v_n = [],[],[],[]
        for t_n in range(1,t_max+1):
            aprob, value = local_model.get_policy_and_value(sess, observation)
            act = np.random.choice(len(actions), p=aprob.flatten())

            observation, r_t, terminal, info = env.step(actions[act])

            o_n.append(observation)
            a_n.append(act)
            r_n.append(np.clip(r_t, -1, 1))
            v_n.append(value)

            if terminal:
                break

        o_n, a_n, r_n, v_n = map(np.array,[o_n,a_n,r_n,v_n])

        ep_rwd += np.sum(r_n)
        T += t_n
        t += t_n

        if terminal:
            R = 0
        else:
            R = local_model.get_value(sess, observation)

        disc_r = discount_rewards(r_n, gamma, env_name, initial=R)

        a_n_one_hot = np.eye(len(actions))[a_n]
        adv = disc_r - v_n
        lr = anneal_learning_rate(T)
        global_model.update(sess, o_n, a_n_one_hot, disc_r, adv, lr)

        if terminal:
            avg_reward = ep_rwd if avg_reward is None\
                                else 0.99*avg_reward + 0.01*ep_rwd
            
            if t_id == 0:
                print "Episode %d, Avg Reward %1.3f and Episode Reward %1.3f"\
                                                    % (n_e, avg_reward, ep_rwd)
                #print "probs: ", aprob, " value: ", value

            global_model.record_summary(sess, writer, avg_reward, ep_rwd, T)
            ep_rwd = 0
            terminal = False
            observation = env.reset()
            n_e += 1

        if t_id == 0 and t - prev_t_save >= SAVE_LOG_INTERVAL:
            prev_t_save += SAVE_LOG_INTERVAL
            saver.save(sess, LOGDIR_TRAIN+'model', T)

        if t_id == 0 and t - prev_t >= PERFORMANCE_LOG_INTERVAL:
            prev_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - start_time
            steps_sec = T/elapsed_time
            print "Performance: Steps %d in %d sec, %d steps per sec, %.2fM steps an hour"\
                            %(T, elapsed_time, steps_sec, steps_sec*(3600/1e6))


def train(sess, global_model):
    writer = tf.train.SummaryWriter(LOGDIR_TRAIN, graph=sess.graph)
    saver = tf.train.Saver()
    envs = [gym.make(FLAGS.env) for i in range(NUM_THREADS)]
    #a3c_thread(0, envs[0], sess, global_model, writer)
    threads = [threading.Thread(target=a3c_thread, args=(i, envs[i],
                                                                sess,
                                                                global_model,
                                                                writer,
                                                                saver))
                                                for i in range(NUM_THREADS)]
    #sess.run(tf.initialize_all_variables())
    for th in threads:
        th.start()

    for th in threads:
        th.join()
        print "A3c thread finished"

    writer.flush()
    writer.close()

def evaluation(sess, model):
    pass

def main(args):
    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)
    global_model = TwoLayerNetA3C(4,2)
    initialize_uninit_var(sess)

    if FLAGS.train:
        train(sess, global_model)
    else:
        evaluation(sess, global_model)

if __name__ == "__main__":
    tf.app.run()