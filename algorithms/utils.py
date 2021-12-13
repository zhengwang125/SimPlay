import random
import numpy as np
import tensorflow as tf
import pickle
#import copy
from time import time

class play2vec():
    def __init__(self, path):
        self.embed_seq, self.initial_state = [], []
        self.rnn_size = 50
        self.embed_size = 20
        self.num_layers = 2
        self.init = tf.constant(0.0, shape=[self.num_layers, 2, 1, self.rnn_size], dtype=tf.float32)
        self.get_encoder_layer()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = path + "model_1/trained_model.ckpt" 
        new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
        new_saver.restore(self.sess, tf.train.latest_checkpoint(path + "model_1"))#
        self.embed_mat = pickle.load(open(path+'embed_mat', 'rb'), encoding='bytes')
        #self.m1 = play2vec_none(path)
        print('Done loading play2vec')

    def get_encoder_layer(self):
        '''
        build Encoder layer
        para:
        - embed_seq: embedding sequence
        - rnn_size: rnn hidden layer num
        - num_layers: rnn cell num
        - source_sequence_length: sequence length of source data
        - encoding_embedding_size: embedding size
        '''
        # Encoder embedding
        self.embed_seq = tf.placeholder(tf.float32, [1, None, self.embed_size])
        
        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
        state_placeholder = tf.placeholder_with_default(self.init, [self.num_layers, 2, 1, self.rnn_size])
        l = tf.unstack(state_placeholder, axis=0)
        self.initial_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                for idx in range(self.num_layers)])
        self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(cell, self.embed_seq, initial_state=self.initial_state,
                                                                    dtype=tf.float32)

    def submit(self, embed, init_state=None):
        if type(embed) == int:
            embed = [embed]
        embed_batch = np.array([[self.embed_mat[i] for i in embed]]).reshape(1,len(embed),-1)
        if init_state == None:
            output, final_state = self.sess.run([self.encoder_output, self.encoder_state], feed_dict={self.embed_seq:embed_batch})
        else:
            output, final_state = self.sess.run([self.encoder_output, self.encoder_state], feed_dict={self.embed_seq:embed_batch, self.initial_state:init_state})
        return output, final_state[-1].h, final_state

class play2vec_none():
    def __init__(self, path):
        self.embed_seq, self.initial_state = [], []
        self.rnn_size = 50
        self.embed_size = 20
        self.num_layers = 2
        self.get_encoder_layer()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = path + "model_1/trained_model.ckpt" 
        new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
        new_saver.restore(self.sess, tf.train.latest_checkpoint(path + "model_1"))#
        self.embed_mat = pickle.load(open(path+'embed_mat', 'rb'), encoding='bytes')
        print('Done loading play2vec_none')
        

    def get_encoder_layer(self):
        '''
        build Encoder layer
        para:
        - embed_seq: embedding sequence
        - rnn_size: rnn hidden layer num
        - num_layers: rnn cell num
        - source_sequence_length: sequence length of source data
        - encoding_embedding_size: embedding size
        '''
        # Encoder embedding
        self.embed_seq = tf.placeholder(tf.float32, [1, None, self.embed_size])
        
        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell
        with tf.variable_scope('scope_2'):
            cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(cell, self.embed_seq, initial_state=None,
                                                                        dtype=tf.float32)
    
def pop_random(seed=300):
    cand = random.randrange(0, seed)
    query = random.randrange(0, seed)
    return (cand,query)

def corrupt_noise(traj, rate_noise, factor):
    new_traj={}
    for count, key in enumerate(traj):
        if count%500==0:
            print('count:',count)
        new_traj[key] = traj[key]
        for i in range(len(traj[key])):
                #adding gauss noise
                for col in range(46):
                    seed = random.random()
                    if seed < rate_noise:
                        new_traj[key][i][col] = traj[key][i][col] + factor * random.gauss(0,1)
    return new_traj

def corrupt_drop(traj, rate_drop):
    new_traj={}
    drop_id={}
    for count, key in enumerate(traj):
        if count%500==0:
            print('count:',count)
        new_traj[key] = traj[key]
        droprow = []
        for i in range(len(traj[key])):
            seed = random.random()
            if seed < rate_drop:
                #dropping
                droprow.append(i)
        new_traj[key] = np.delete(new_traj[key], droprow, axis = 0)
        drop_id[key]=droprow
    return drop_id #new_traj
