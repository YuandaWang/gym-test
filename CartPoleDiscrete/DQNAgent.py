
# This deepQnetwork class is designed for gym tests

import tensorflow as tf
import numpy as np
import random

class DeepQNetwork():
    def __init__(self, observe_size, action_size):
        
        self.ob_input = tf.placeholder(tf.float32, shape = [None, observe_size])
        
        # Network Parameters
        fc1_num = 20

        # full connneted
        self.W1 = tf.Variable(tf.random_normal([observe_size, fc1_num]))
        self.b1 = tf.Variable(tf.random_normal([fc1_num]))
        self.fc1 = tf.nn.relu(tf.matmul(self.ob_input, self.W1) + self.b1)
        
        self.W2 = tf.Variable(tf.random_normal([fc1_num, action_size]))
        self.b2 = tf.Variable(tf.random_normal([action_size]))
        # Out
        self.Qout = tf.matmul(self.fc1, self.W2) + self.b2
        self.predict = tf.argmax(self.Qout, 1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.updateModel = self.trainer.minimize(self.loss)
        

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])