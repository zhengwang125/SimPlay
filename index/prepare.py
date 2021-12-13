# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:16:11 2021

@author: wang_zheng
"""
import pickle
from rl_main_skip import Subtraj, DeepQNetwork
import time
from utils import play2vec, pop_random
import tensorflow as tf
import numpy as np
from sortedcontainers import SortedList

tf.reset_default_graph()

def prepare():
    qtrainsize, qtestsize, dsize = 100, 100, 500
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    traj_tokens=sorted(traj_tokens,key=lambda i:len(i))[-4*dsize-qtrainsize-qtestsize:]
    import random
    random.seed(0)
    random.shuffle(traj_tokens)
    return qtrainsize, qtestsize, dsize, traj_tokens

def exact(traj_c, traj_q):
    Q_outputs, Q_state, _ = m0.submit(traj_q)
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    for i in range(len(traj_c)):
         C_outputs, C_state, _ = m0.submit(traj_c[i:])
         for j in range(C_outputs.shape[1]):
             #print('sub-range:', [i, i+j])
             temp = np.linalg.norm(Q_state - C_outputs[0,j,:])
             contain.add(temp)
             if temp < subsim:
                 subsim = temp
                 subtraj = [i, i+j]
    return subsim, subtraj

path = r'./SoccerData/'
path1 = r'./SoccerData/index_data/'
m0 = play2vec(path)
    
################################# THIS IS FOR GROUND #########################################################################################    
qtrainsize, qtestsize, dsize, traj_tokens = prepare()
#(dsize, None), (2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)

for (a, b) in [(dsize, None), (2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)]:
    print('contain in ground', (a,b))
    query_train_set, query_test_set = traj_tokens[0:qtrainsize], traj_tokens[qtrainsize:qtrainsize+qtestsize]
    if b == None:
        cand_set = traj_tokens[-a:]
    else:
        cand_set = traj_tokens[-a:-b]

    path_ground = r'./SoccerData/index_data/ground_'+str(a)+'_'+str(b)+'/'
    
    start = time.time()
    SUBsim, SUBtraj = [[] for i in range(qtestsize)], [[] for i in range(qtestsize)]
    
    for i in range(len(query_test_set)):
        if i % 10 == 0:
            print('process', i, '/', len(query_test_set))
        contain = SortedList()
        for j in range(len(cand_set)):
            subsim, subtraj = exact(cand_set[j], query_test_set[i])
            SUBsim[i].append(subsim)
            SUBtraj[i].append(subtraj)
            pickle.dump(contain, open(path_ground+'contain_'+str(i), 'wb'), protocol=2)
    pickle.dump(SUBsim, open(path_ground+'SUBsim', 'wb'), protocol=2)
    pickle.dump(SUBtraj, open(path_ground+'SUBtraj', 'wb'), protocol=2)
    print('time cost of ground', (a, b), time.time()-start)

path_ground_pre = r'./SoccerData/index_data/ground_'+str(dsize)+'_'+str(None)+'/'
for (a, b) in [(2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)]:
    print('refine in ground', (a,b))
    query_train_set, query_test_set = traj_tokens[0:qtrainsize], traj_tokens[qtrainsize:qtrainsize+qtestsize]
    path_ground = r'./SoccerData/index_data/ground_'+str(a)+'_'+str(b)+'/'
    for i in range(len(query_test_set)):
        tmp = pickle.load(open(path_ground+'contain_'+str(i), 'rb'), encoding='bytes') + pickle.load(open(path_ground_pre+'contain_'+str(i), 'rb'), encoding='bytes')
        pickle.dump(tmp, open(path_ground+'contain_'+str(i), 'wb'), protocol=2)
    
    SUBsim = np.hstack((pickle.load(open(path_ground+'SUBsim', 'rb'), encoding='bytes'), pickle.load(open(path_ground_pre+'SUBsim', 'rb'), encoding='bytes'))).tolist()
    SUBtraj = np.hstack((pickle.load(open(path_ground+'SUBtraj', 'rb'), encoding='bytes'), pickle.load(open(path_ground_pre+'SUBtraj', 'rb'), encoding='bytes'))).tolist()
    pickle.dump(SUBsim, open(path_ground+'SUBsim', 'wb'), protocol=2)
    pickle.dump(SUBtraj, open(path_ground+'SUBtraj', 'wb'), protocol=2)
    path_ground_pre = path_ground
##################################################################################################################################################
