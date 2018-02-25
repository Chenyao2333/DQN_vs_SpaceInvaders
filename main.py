#! /usr/bin/env python3

import gym
import tensorflow as tf
import numpy as np
import numpy.random
import random
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"


COLOR_ID = {0: 1, 3284960: 2, 10568276: 3, 8747549: 4, 11790730: 5, 5224717: 6, 9269902: 7, 9825272: 8}
COLOR_CNT = 8

SKIP_FRAMES = 3
LR = 0.01
Y = 0.95
E = 0.05

def findCOLOR_ID():
    global COLOR_CNT, COLOR_ID
    env = gym.make("SpaceInvaders-v0")
    
    for i in range(10):
        s = env.reset()
        for j in range(100):
            a = env.action_space.sample()
            s, _, d, _ = env.step(a)
            print(s.shape)
            #env.render()
            if d:
                break
            for row in s:
                for c in row:
                    int_c = c[0]*255*255 + c[1]*255 + c[2]
                    if int_c not in COLOR_ID:
                        COLOR_CNT += 1
                        COLOR_ID[int_c] = COLOR_CNT
        # Total 7 different COLOR_ID in games
        #if len(COLOR_ID) >= 7:
        #    break

    print("COLOR_ID: ", COLOR_ID)
    print("COLOR_CNT: ", len(COLOR_ID))


def RGB2ColorID(s):
    ret = np.zeros(shape=[210, 160])
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            c = s[i][j]
            int_c = c[0]*255*255 + c[1]*255 + c[2]
            ret[i][j] = COLOR_ID[int_c]
    
    return ret


class DQN(object):
    def __init__(self):
        self.__version__ = "0.0.4"
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, SKIP_FRAMES, 210, 160])
        self.batch_size = tf.shape(self.inputs)[0]

        self.onehot = tf.one_hot(self.inputs, 8, dtype=tf.float16)
        self.reshape = tf.reshape(self.onehot, [self.batch_size, 210, 160, SKIP_FRAMES * 8])
        self.conv1 = tf.layers.conv2d(self.reshape, filters = 16, strides = (4, 4), kernel_size = (5, 5), padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(self.conv1, filters = 32, strides = (4, 4), kernel_size = (3, 3), padding="same", activation=tf.nn.relu)
        self.flat = tf.layers.flatten(self.conv2)
        self.dense1 = tf.layers.dense(self.flat, 256, activation=tf.nn.relu)
        self.dense2 = tf.layers.dense(self.dense1, 128, activation=tf.nn.sigmoid)
        self.logits = tf.layers.dense(self.dense2, 6)
        self.outputs = tf.clip_by_value(self.logits, -10, 100) # clip the rewards

        self.predicts = tf.argmax(self.outputs, 1)

        self.targets = tf.placeholder(dtype=tf.float16, shape=[None,6])
        self.loss = tf.div(tf.reduce_sum(tf.square(self.targets - self.outputs)), tf.cast(self.batch_size, tf.float16))

        self.gd = tf.train.GradientDescentOptimizer(learning_rate=LR)
        self.optimize = self.gd.minimize(self.loss)


        #tf.summary.histogram("loss", self.loss)
        #self.merged = tf.summary.merge_all()


class ReplyBuffer(object):
    def __init__(self, N = 100 * 100 * 100):
        self.N = N
        self.frames = [] # one item in frames constituting by SKIP_FRAMES images
        self.actions = []
        self.rewards = []
        self.dones = []

    def put(self, frames, action, reward, done):
        self.frames.append(frames)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        self.frames = self.frames[-self.N:]
        self.actions = self.actions[-self.N:]
        self.rewards = self.rewards[-self.N:]
        self.dones = self.dones[-self.N:]
    
    def sample(self, batch_size = 32):
        tot = len(self.actions) - 1

        if tot < batch_size:
            batch_size = tot
            return None
        
        choices = []
        while len(choices) < batch_size:
            i = random.randint(0, tot - 1)
            if i not in choices:
                choices.append(i)


        ret = []

        for i in choices:
            s = {}
            s["frames"] = self.frames[i]
            s["action"] = self.actions[i]
            s["reward"] = self.rewards[i]
            s["next_frames"] = self.frames[i + 1]
            s["done"] = self.dones[i]

            ret.append(s)

        return ret

def reset(env):
    frames = [RGB2ColorID(env.reset())]

    for i in range(SKIP_FRAMES-1):
        s1, _, _, _ = env.step(0)
        s1 = RGB2ColorID(s1)
        frames.append(s1)

    return frames

def cal_action(sess, env, net, f0, frames_cnt, training=False):
    e = 0.05
    if training:
        e = np.max([0.1, 1 - frames_cnt/(100*100*100) * 0.9])
    
    if random.random() < e:
        print("rand action: ")
        return env.action_space.sample()

    predicts = sess.run(net.predicts, feed_dict={net.inputs: [f0]})
    print("predict action: ", predicts[0])
    return predicts[0]

def step(env, a):
    f1 = []
    reward = 0
    done = False
    scores = 0

    for i in range(SKIP_FRAMES):
        f, r, d, _ = env.step(a)
        f = RGB2ColorID(f)
        env.render()
        if d:
            f1 = []
            reward = -1
            done = True
            return f1, reward, done, 0

        f1.append(f)
        reward += r
    
    scores = reward
    if reward > 0:
        reward = 1
    
    return f1, reward, done, scores

def train(sess, buff, net):
    samples = buff.sample()
    if samples == None:
        return
    
    frames = []
    next_frames = []
    for s in samples:
        frames.append(s["frames"])
        next_frames.append(s["next_frames"])
    
    noutputs = sess.run(net.outputs, feed_dict={net.inputs: next_frames})
    targets = []
    
    for i in range(len(samples)):
        r = samples[i]["reward"]

        if not samples[i]["done"]:
            r += Y*np.argmax(noutputs[i])
        
        a = samples[i]["action"]
        t = np.copy(noutputs[i])
        t[a] = r 

        targets.append(t)

    _, ouputs, loss = sess.run((net.optimize, net.outputs, net.loss), feed_dict={net.inputs: frames, net.targets: targets})
    average_outputs = np.mean(ouputs)
    print("average_outputs: %.3f, loss: %d" % (average_outputs, loss))

def main(argv=None):
    #findCOLOR_ID()

    env = gym.make("SpaceInvaders-v0")
    env.reset()

    net = DQN()
    buff = ReplyBuffer()
    
    saver = tf.train.Saver()

    frames_cnt = 0
    
    with tf.Session() as sess:
        #writer = tf.summary.FileWriter("logdir", sess.graph)
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, "./tf_ckpts/0.0.2/ToyNet_15_195.ckpt")
        i = 0

        while frames_cnt <= 100*100*100*100*10:
            i += 1
            
            f0 = reset(env)
            totalScore = 0

            j = 0
            while j <= 20000:
                j += 1

                a0 = cal_action(sess, env, net, f0, frames_cnt, training=True)
                f1, r0, done, scores = step(env, a0)

                totalScore += scores

                buff.put(f0, a0, r0, done)

                train(sess, buff, net)

                f0 = f1

                frames_cnt += SKIP_FRAMES
                if done:
                    with open("scores.txt") as f:
                        print("Episode: %d, Frames: %d, TotalFrames: %d, Scores: %d" % (i, j * SKIP_FRAMES, frames_cnt, totalScore))
                    break



if __name__ == '__main__':
    tf.app.run()
