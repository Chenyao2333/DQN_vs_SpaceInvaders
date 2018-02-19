#! /usr/bin/env python3

import gym
import tensorflow as tf
import numpy as np
import numpy.random


COLOR_ID = {0: 1, 3284960: 2, 10568276: 3, 8747549: 4, 11790730: 5, 5224717: 6, 9269902: 7, 9825272: 8}
COLOR_CNT = 8

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


class ToyNet(object):
    def __init__(self):
        self.inputs = tf.placeholder(dtype=tf.float16, shape=[None,210,160,1])
        self.conv1 = tf.layers.conv2d(self.inputs, filters = 12, strides = (2, 2), kernel_size = (3, 3), padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(self.conv1, filters = 12, strides = (2, 2), kernel_size = (3, 3), padding="same", activation=tf.nn.relu)
        self.flat = tf.layers.flatten(self.conv2)
        self.dense1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
        self.dense2 = tf.layers.dense(self.dense1, 512, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.dense2, 6)
        self.outputs = tf.clip_by_value(self.logits, -10, 100) # maximal rewards is 100*10

        self.predicts = tf.argmax(self.outputs, 1)

        self.targets = tf.placeholder(dtype=tf.float16, shape=[None,6])
        self.loss = tf.div(tf.reduce_sum(tf.square(self.targets - self.outputs)), tf.cast(tf.shape(self.inputs)[0], tf.float16))

        self.gd = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.optimize = self.gd.minimize(self.loss)


        #tf.summary.histogram("loss", self.loss)
        #self.merged = tf.summary.merge_all()


class Buffer(object):
    def __init__(self, size):
        self.buff = []
        self.size = size
        self.step = 0

    def put(self, input, target):
        self.buff.append((input, target))
        if len(self.buff) > self.size:
            self.buff = self.buff[-self.size:]
    
    def train(self, sess, net):
        inputs = []
        targets = []

        for inp, tar in self.buff:
            #print(inp.shape, tar.shape)
            inputs.append(inp)
            targets.append(tar)

        #inputs = np.reshape(np.array(inputs), [len(self.buff), 210, 160, 1])
        #targets = np.reshape(np.array(targets), [len(self.buff), 6])

        #print(inputs.shape, targets.shape)

        l, _ = sess.run([net.loss, net.optimize], feed_dict={net.inputs: inputs, net.targets: targets})

        self.step += 1
        #writer.add_summary(summary, self.step)
        print("loss: ", l)

def main(argv=None):
    #findCOLOR_ID()

    env = gym.make("SpaceInvaders-v0")
    env.reset()

    # s = []
    # for i in range(20):
    #     a = env.action_space.sample()
    #     s1, _, done, _ = env.step(a)
    #     if done:
    #         break

    #     s.append(RGB2ColorID(s1))

    net = ToyNet()
    buff = Buffer(8)

    lr = 0.98
    y = 0.99
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logdir", sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            s = env.reset()
            s = RGB2ColorID(s)
            s = np.reshape(s, [210, 160, 1])
            totalScore = 0

            for j in range(20000):
                print("====================================")
                print("Round: %d, Steps: %d" % (i, j))
                outputs = sess.run(net.outputs, feed_dict={net.inputs: np.reshape(s, [1, 210, 160, 1])})[0]
                print("outputs: ", outputs)
                a = np.argmax(outputs + np.random.rand(1, 6) * (3)/(i+1))
                print("action: ", a)
                s1, r, done, _ = env.step(a)
                totalScore += r

                # avoid the reward is too big to cause NaN
                r /= 10
                env.render()
                s1 = RGB2ColorID(s1)
                s1 = np.reshape(s1, [210, 160, 1])

                # if it ends the game, we set this step as negtative reward
                if done:
                    r = -5

                next_r = np.max(sess.run(net.outputs, feed_dict={net.inputs: np.reshape(s1, [1, 210, 160, 1])})[0])
                final_r = (1-lr)*outputs[a] + lr*(r + y*next_r)

                print("next_r: ", next_r)
                print("final_r: ", final_r)

                targets = np.copy(outputs)
                targets[a] = final_r

                if abs(final_r - outputs[a]) >0.1:
                    buff.put(s, targets)
                    buff.train(sess, net)

                if done:
                    break

                s = s1

            with open("scores.txt", "wa") as f:
                f.write("Eposide: %d, Score: %d\n" % (j, totalScore))



if __name__ == '__main__':
    tf.app.run()