import numpy as np
import pickle
from utils import play2vec
import tensorflow as tf

path = r'./SoccerData/'
tf.reset_default_graph()

m0 = play2vec(path)

class Subtraj():
    def __init__(self, cand_train, query_train):
        self.action_space = ['0', '1', '2', '3','4']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.cand_train_name = cand_train
        self.query_train_name = query_train
        self.presim = 0
        self.RW = 0.0
        self.delay = 0
        self._load()

    def _load(self):
        self.cand_train_data = pickle.load(open(self.cand_train_name, 'rb'), encoding='bytes')
        self.query_train_data = pickle.load(open(self.query_train_name, 'rb'), encoding='bytes')
        
    def reset(self, episode, label='E'):
        # prefix_state --> [split_point, index]
        # suffix_state --> [index + 1, len - 1]
        # return observation
        _, self.query_state_data, _ = m0.submit(self.query_train_data[episode])        
        self.split_point = 0
        self.init_state = None
        self.init_state_real = None
        _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][0:1], self.init_state)
        self.length = len(self.cand_train_data[episode])
        
        self.presim = np.linalg.norm(self.query_state_data[-1] - self.h0[-1])
        observation = np.array([self.presim, self.presim]).reshape(1,-1)
        self.subsim = min(self.presim, self.presim)
        self.subtraj = [0, 0]
        if label == 'T':
            self.subsim_real = min(self.presim, self.presim)
        
        return observation, self.length, -1        
    
    def step(self, episode, action, index, label='E'):
        if action == 0: #non-split
            if index == self.length - 1:
                done = True
            else:
                done = False    
            #state transfer
            _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state)
            self.presim = np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
                
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
            
            if label == 'T':
                _, self.h0_real, self.init_state_real = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state_real)
                self.presim_real = np.linalg.norm(self.query_state_data[-1] -  self.h0_real[-1])
                last_subsim = self.subsim_real
                self.subsim_real = min(last_subsim, self.presim_real)
                self.RW = last_subsim - self.subsim_real              
            return observation, self.RW, done, -1
           
        if action == 1: #split
            if index == self.length - 1:
                done = True
            else:
                done = False   
            self.split_point = index
            self.init_state = None
            _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state)
            
            #state transfer
            self.presim = np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, index]
                
            if label == 'T':
                self.init_state_real = None
                _, self.h0_real, self.init_state_real = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state_real)
                self.presim_real = np.linalg.norm(self.query_state_data[-1] -  self.h0_real[-1])
                last_subsim = self.subsim_real
                self.subsim_real = min(last_subsim, self.presim_real)
                self.RW = last_subsim - self.subsim_real
            
            return observation, self.RW, done, -1
        
        if action > 1: #skipping
            #state transfer
            INX = min(index + action - 1, self.length - 1)
            if INX == self.length - 1:
                done = True
            else:
                done = False
                
            _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][INX:INX+1], self.init_state)
            self.presim = np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [self.split_point, INX]
            
            if label == 'T':
                _, self.h0_real, self.init_state_real = m0.submit(self.cand_train_data[episode][index:INX+1], self.init_state_real)
                self.presim_real = np.linalg.norm(self.query_state_data[-1] -  self.h0_real[-1])
                last_subsim = self.subsim_real
                self.subsim_real = min(last_subsim, self.presim_real)
                self.RW = last_subsim - self.subsim_real           
            return observation, self.RW, done, INX
            
    def output(self, index, episode, label='E'):
        if label == 'check':
            print('check', self.subsim, self.subtraj, self.subsim_real)
        if label == 'E':
            _, hidden,  _ = m0.submit(self.cand_train_data[episode][self.subtraj[0]:self.subtraj[1]+1], None)
            self.subsim_real = np.linalg.norm(self.query_state_data[-1] -  hidden[-1])
        return [self.subsim_real, self.subtraj]