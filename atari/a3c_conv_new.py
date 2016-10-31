from a3c_trainer import A3CTrainer
from rmsprop_applier import RMSPropApplier
from model import ConvNetA3C
import tensorflow as tf
import threading
import numpy as np
import time

ENV_GYM = 'Pong-v0'
ENV_ALE = 'pong'
WIDTH = 84
HEIGHT = 84
NUM_FRAMES = 4
MAX_TIME_STEPS = 10**8
NUM_ACTIONS = 3
NUM_THREADS = 8
LOGDIR_TRAIN = ENV_GYM.strip('-v0') + "-a3c_train_logs/"
LR_LOW = 1e-4
LR_HIGH = 1e-2
RATE = 0.4226
LOCAL_T_MAX = 5
GAMMA = 0.99
SAVE_INTERVAL = 50000

global_t = 0

def log_uniform(lo, hi, rate):
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return np.exp(v)

def run_a3c_thread(t_id, sess, a3c_trainers, writer, saver):
    global global_t

    a3c_trainer = a3c_trainers[t_id]
    t_max = LOCAL_T_MAX
    a3c_trainer.start_time = time.time()
    prev_t = 0
    while global_t < MAX_TIME_STEPS:
        diff = a3c_trainer.run_steps(sess, t_max, global_t, writer)
        global_t += diff

        if t_id == 0 and global_t - prev_t >= SAVE_INTERVAL:
            prev_t += SAVE_INTERVAL
            saver.save(sess, LOGDIR_TRAIN + 'model.ckpt')

def main():
    global_model = ConvNetA3C(HEIGHT, WIDTH, NUM_FRAMES, NUM_ACTIONS, summary=True)
    lr_ph = tf.placeholder(dtype=tf.float32, name='lr')
    grad_applier = RMSPropApplier(learning_rate = lr_ph,
                            decay = 0.99,
                            momentum = 0.0,
                            epsilon = 0.1,
                            clip_norm = 40.0,
                            device = "/cpu:0")

    # build all models in the main thread. creating model graphs
    # within each thread causing some non-trivial tensorflow issues
    init_lr = log_uniform(LR_LOW, LR_HIGH, RATE)
    model_dim = (HEIGHT, WIDTH, NUM_FRAMES, NUM_ACTIONS)
    a3c_trainers = [A3CTrainer(i, ENV_ALE, global_model, init_lr,lr_ph,
                    grad_applier, MAX_TIME_STEPS, model_dim, GAMMA)
                    for i in range(NUM_THREADS)]

    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    writer = tf.train.SummaryWriter(LOGDIR_TRAIN, graph=sess.graph)
    saver = tf.train.Saver()

    threads = [threading.Thread(target=run_a3c_thread, args=(i,
                                                            sess,
                                                            a3c_trainers,
                                                            writer, saver,
                                                            ))
                                        for i in range(NUM_THREADS)]


    for th in threads:
        th.start()

    for th in threads:
        th.join()
        print "A3C finished"

if __name__ == "__main__":
    main()
