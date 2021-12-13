import numpy as np
import pickle
from utils import play2vec
import tensorflow as tf

path = r'./SoccerData/'
tf.reset_default_graph()

m0 = play2vec(path)

class Subtraj():
    def __init__(self, cand_train, query_train, Sourceint_Mapback, cand_test_index, SUBSIM_SUBSET):
        self.action_space = ['0', '1']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.cand_train_name = cand_train
        self.query_train_name = query_train
        self.presim = 0
        self.RW = 0.0
        self.delay = 0
        self.Sourceint_Mapback = Sourceint_Mapback
        self.cand_test_index = cand_test_index
        self.SUBSIM_SUBSET = SUBSIM_SUBSET
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
        _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][0:1], self.init_state)
        self.length = len(self.cand_train_data[episode])
        
        S = self.Sourceint_Mapback[self.cand_test_index[episode]][0]
        
        self.presim = self.SUBSIM_SUBSET[episode][(S, S)] #np.linalg.norm(self.query_state_data[-1] - self.h0[-1])
        
        observation = np.array([self.presim, self.presim]).reshape(1,-1)
        self.subsim = min(self.presim, self.presim)
        self.subtraj = [0, 0]
        if label == 'T':
            self.subsim_real = min(self.presim, self.presim)
        return observation, self.length
      
        
    def step(self, episode, action, index, label='E'):
        if action == 0: #non-split 
            #state transfer
            _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state)
            S = self.Sourceint_Mapback[self.cand_test_index[episode]][self.split_point]
            E = self.Sourceint_Mapback[self.cand_test_index[episode]][index]
            
            self.presim = self.SUBSIM_SUBSET[episode][(S, E)] #np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
            
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
                            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [S, E]
            
            if label == 'T':
                last_subsim = self.subsim_real
                self.subsim_real = min(last_subsim, self.presim)
                self.RW = last_subsim - self.subsim_real      
                    
            return observation, self.RW
        if action == 1: #split
            self.split_point = index
            self.init_state = None
            _, self.h0, self.init_state = m0.submit(self.cand_train_data[episode][index:index + 1], self.init_state)
            
            S = self.Sourceint_Mapback[self.cand_test_index[episode]][self.split_point]
            E = self.Sourceint_Mapback[self.cand_test_index[episode]][index]
            
            #state transfer
            self.presim = self.SUBSIM_SUBSET[episode][(S, E)] #np.linalg.norm(self.query_state_data[-1] -  self.h0[-1])
            observation = np.array([self.subsim, self.presim]).reshape(1,-1)
            
            
            if self.presim < self.subsim:
                self.subsim = self.presim
                self.subtraj = [S, E]
                
            if label == 'T':
                last_subsim = self.subsim_real                
                self.subsim_real = min(last_subsim, self.presim)
                self.RW = last_subsim - self.subsim_real
            
            return observation, self.RW

    def output(self):
        return [self.subsim, self.subtraj]