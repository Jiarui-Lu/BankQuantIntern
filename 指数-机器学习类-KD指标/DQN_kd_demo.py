# coding: utf-8
from __future__ import division

import tensorflow as tf
import numpy as np
import random
from collections import deque
# from yahoo_finance import Share
import datetime
import matplotlib.pyplot as plt
import cv2
import time
import matplotlib.image as pimg
import pandas as pd
import KD_draw
# In[2]:

KD_draw

day_len = 15   # 每筆資料的日期天數


# In[3]:
# print("2330 start to analyze")
# stock = Share('2330.TW')
startDay = '2015-07-14'
today = datetime.date.today()
stock_data=pd.read_excel(r'data\2330.xls',index_col=0)[:50]

print("Historical data since", startDay,": ", len(stock_data))

my_train = np.zeros((len(stock_data) - day_len, day_len), dtype=np.float)
# my_train
my_img = np.zeros((len(my_train), 64, 64), dtype=np.float)


# my_img


# ### 設定訓練資料

# In[6]:
def normalize(li):  # a list
    mean = sum(li) / float(len(li))
    for i in range(0, len(li)):
        li[i] = li[i] - mean
    return li


for i in range(0, len(my_train)):
    for j in range(0, day_len):
        my_train[i, j] = float(stock_data['Close'][stock_data.index[i]])

label_train = np.zeros(len(my_train), dtype=np.float)

# set reward
for x in range(0, len(my_train) - 1):
    label_train[x] = my_train[x + 1][day_len - 1] - my_train[x][day_len - 1]


np.set_printoptions(threshold=np.inf)  # print full array
perf_collect = {}


def img_gray(my_img):
    new_img = np.zeros((64, 64), dtype=np.float)
    for x in range(0, len(my_img)):
        for y in range(0, len(my_img[x])):
            if (my_img[x, y, 0] != 1 and my_img[x, y, 1] != 1 and my_img[x, y, 2] != 1):
                new_img[x, y] = 0
            elif (my_img[x, y, 0] != 1 and my_img[x, y, 2] != 1):
                new_img[x, y] = 0.7
            elif (my_img[x, y, 0] != 1 and my_img[x, y, 1] != 1):
                new_img[x, y] = 0.3
            else:
                new_img[x, y] = 1
    return new_img


for x in range(0, len(my_train) - 1):  # load file
    img = pimg.imread(r'result\KD_{}.png'.format(str(x)))
    img2 = cv2.resize(img, (64, 64))
    my_img[x] = img_gray(img2)

# used for cross-validation
cross = 30
my_test = my_img[cross:, ]
my_img = my_img[:cross, ]

label_test = label_train[cross:, ]
label_train = label_train[:cross, ]

x = open(r"result\2330KD_1pic.txt", "w")
x.close()
fo = open(r"result\2330KD_1pic.txt", "a")


# In[8]:

class TWStock():
    def __init__(self, stock_data, label):
        self.stock_data = stock_data
        self.stock_index = 0
        self.label = label

    def render(self):
        # 尚未實作
        return

    def reset(self):
        self.stock_index = 0
        return self.stock_data[0]

    # 0: 觀望, 1: 持有多單, 2: 持有空單
    def step(self, action):
        self.stock_index += 1
        # action_reward = self.stock_data[self.stock_index][day_len-1] - self.stock_data[self.stock_index+10][day_len-1]
        action_reward = self.label[self.stock_index]
        if (action == 0):
            action_reward = 0

        if (action == 2):
            action_reward = -1 * action_reward
        # print(str(action)+" "+str(action_reward))

        cv2.imshow('image_mean2', self.stock_data[self.stock_index])
        cv2.waitKey(1)

        stock_done = False
        if self.stock_index + 10 >= len(self.stock_data) - 1:
            stock_done = True
        else:
            stock_done = False
        return self.stock_data[self.stock_index], action_reward, stock_done, 0


# In[9]:

def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# ### 載入資料

# In[10]:

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n

        self.state_dim = day_len
        self.action_dim = 3

        self.create_Q_network()
        self.create_training_method()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_Q_network(self):
        # -----------------------end cnn   start

        '''W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2
        '''
        # network weights
        W_conv1 = self.weight_variable([8, 8, 1, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1024, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.action_dim])
        b_fc2 = self.bias_variable([self.action_dim])

        # input layer

        self.state_input = tf.placeholder("float", [None, 64, 64])
        input1 = tf.reshape(self.state_input, [-1, 64, 64, 1])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(input1, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        print("dimension:", h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1024])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # Q Value layer
        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

        # tf.scalar_summary("cost", values=self.cost)
        # tf.histogram_summary("cost", values=self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        # print(reward_batch)
        next_state_batch = [data[3] for data in minibatch]
        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        # save network every 100 iteration

    # if self.time_step % 50 == 0:
    # self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------


# ## main function

# In[11]:

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 15  # Episode limitation
STEP = 2  # 300 # Step limitation in an episode
TEST = 2  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)
    env = TWStock(my_img, label_train)
    agent = DQN(env)

    print('開始執行')
    train_output = ""
    rate_string = ""
    for episode in range(EPISODE):

        # initialize task
        state = env.reset()

        # Train
        out = "train\n"

        train_reward = 0
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for trai

            next_state, reward, done, _ = env.step(action)
            out += str(reward) + " "
            train_reward += reward
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        anal = out.split()
        p = 0.0
        n = 0.0
        for x in range(1, len(anal) - 1):
            if (float(anal[x]) > 0):
                p += float(anal[x])
            elif (float(anal[x]) < 0):
                n += float(anal[x])
        try:
            rate = round(p / (n * (-1) + p), 2)
            rate_string += str(rate) + " "
            fo.write(out + "\n")
            train_output += str(train_reward) + " "
            # Test every 100 episodes
            if episode % 10 == 0:
                out = "test\n"
                env1 = TWStock(my_test, label_test)
                total_reward = 0

                for i in range(TEST):
                    state = env1.reset()

                    for j in range(STEP):
                        env1.render()
                        action = agent.action(state)  # direct action for test
                        state, reward, done, _ = env1.step(action)
                        out += str(action) + " " + str(reward) + ","
                        total_reward += reward
                        if done:
                            break
                fo.write(out + "\n")
                print(train_output)
                train_output = ""
                print('episode: ', episode, 'Total Return:', total_reward, 'training Rate past10:', rate_string)
                rate_string = ""
        except:
            pass
    print('程式結束')
    df=pd.DataFrame([total_reward],columns=['Total Return'])
    df.to_excel(r'result\total return.xls')


if __name__ == '__main__':
    main()

