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
import torch
from train_index import NeuNet, par, obtain_embeds
import heapq

args = par()
tf.set_random_seed(0)
tf.reset_default_graph()

def prepare():
    qtrainsize, qtestsize, dsize = 100, 100, 500
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    traj_tokens=sorted(traj_tokens,key=lambda i:len(i))[-4*dsize-qtrainsize-qtestsize:]
    import random
    random.seed(0)
    random.shuffle(traj_tokens)
    return qtrainsize, qtestsize, dsize, traj_tokens

def run_evaluate_skip(choose, qsize_ = 100, dsize_ = 500):
    e = 0
    similarity = [[] for i in range(qsize_)]
    for q in range(qsize_):
        for d in range(dsize_):
            if d >= choose:
                break
            observation, steps, INX = env.reset(e, 'E')
            for index in range(1, steps):
                if index <= INX:
                    continue
                action = RL.fast_online_act(observation)
                observation_, _, done, INX  = env.step(e, action, index, 'E')
                observation = observation_
            res = env.output(index, e)
            similarity[q].append(res[0])
            e += 1
            #print('e',e)
    return similarity

path = r'./SoccerData/'
path1 = r'./SoccerData/index_data/'
m0 = play2vec(path)
checkpoint = r'./SoccerData/index_data/model/your_model.pth'
#################################### THIS IS FOR RLS-SKIP LEARN INDEX #########################################
qtrainsize, qtestsize, dsize, traj_tokens = prepare()
pruning = 0.8
##FOR TESTING DATA LEARN INDEX, EFFICIENCY
query_test_set = traj_tokens[qtrainsize:qtrainsize+qtestsize]
for (a, b) in [(dsize, None), (2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)]:
    print('contain in learned index', (a,b))
    if b == None:
        cand_set = traj_tokens[-a:]
    else:
        cand_set = traj_tokens[-a:-b]
    choose = round(dsize*(1-pruning))
    NN = NeuNet(args)
    NN.load_state_dict(torch.load(checkpoint))
    
    cand_embed = obtain_embeds(cand_set)
    cand_index_embed = NN(torch.tensor(cand_embed), 'test')
    
    start = time.time()
    query_test_set_embed = obtain_embeds(query_test_set)
    query_test_set_index_embed = NN(torch.tensor(query_test_set_embed), 'test')

    query_test, cand = [], []
    for i in range(qtestsize):
        qembed = query_test_set_index_embed[i]
        nums, similarity_ = [], []
        for j in range(dsize):
            cembed = cand_index_embed[j]
            nums.append((torch.dist(qembed, cembed, p=2), j))
        nums = heapq.nsmallest(choose, nums)
        for num in nums:
            query_test.append(query_test_set[i])
            cand.append(cand_set[num[1]])
                
    env = Subtraj(cand, query_test)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/skip/your_model.h5')
    path_learn = r'./SoccerData/index_data/learn_'+str(a)+'_'+str(b)+'/'
    similarity = run_evaluate_skip(choose, qtestsize, dsize)
    pickle.dump(similarity, open(path_learn+'similarity-learn', 'wb'), protocol=2)
    print('time cost of rls-skip learn index', (a, b), time.time()-start)

##refine save stuff
path_learn_pre = r'./SoccerData/index_data/learn_'+str(dsize)+'_'+str(None)+'/'
for (a, b) in [(2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)]:
    print('refine in ground', (a,b))
    query_train_set, query_test_set = traj_tokens[0:qtrainsize], traj_tokens[qtrainsize:qtrainsize+qtestsize]
    path_learn = r'./SoccerData/index_data/learn_'+str(a)+'_'+str(b)+'/'
    tmp = np.hstack((pickle.load(open(path_learn+'similarity-learn', 'rb'), encoding='bytes'), pickle.load(open(path_learn_pre+'similarity-learn', 'rb'), encoding='bytes'))).tolist()
    pickle.dump(tmp, open(path_learn+'similarity-learn', 'wb'), protocol=2)
    path_learn_pre = path_learn

#EFFECTIVENESS
for (a, b) in [(dsize, None), (2*dsize, dsize), (3*dsize, 2*dsize), (4*dsize, 3*dsize)]:
    path_learn = r'./SoccerData/index_data/learn_'+str(a)+'_'+str(b)+'/'
    path_ground = r'./SoccerData/index_data/ground_'+str(a)+'_'+str(b)+'/'
    similarity = pickle.load(open(path_learn+'similarity-learn', 'rb'), encoding='bytes')
    res_ap, res_mr, res_rr = [], [], []
    for q in range(qtestsize):
        SUBsim = pickle.load(open(path_ground+'contain_'+str(q), 'rb'), encoding='bytes')
        t = SUBsim.index(min(similarity[q])) + 1
        res_ap.append(min(similarity[q])/SUBsim[0])
        res_mr.append(t)
        res_rr.append(t/len(SUBsim))
    print('effectiveness of learn idnex', np.mean(res_ap), np.mean(res_mr), np.mean(res_rr))

##################################THIS IS FOR RLS-SKIP LEARN INDEX (vary prunning) ################################################################################    
qtrainsize, qtestsize, dsize, traj_tokens = prepare()
##FOR TESTING DATA LEARN INDEX, EFFICIENCY
query_test_set = traj_tokens[qtrainsize:qtrainsize+qtestsize]
for pruning in [0.5, 0.6, 0.7, 0.8]:
    print('contain in learn index', pruning)
    cand_set = traj_tokens[-dsize:]
    
    choose = round(dsize*(1-pruning))
    NN = NeuNet(args)
    NN.load_state_dict(torch.load(checkpoint))
    cand_embed = obtain_embeds(cand_set)
    cand_index_embed = NN(torch.tensor(cand_embed), 'test')
    
    start = time.time()
    query_test_set_embed = obtain_embeds(query_test_set)
    query_test_set_index_embed = NN(torch.tensor(query_test_set_embed), 'test')
    query_test, cand = [], []
    for i in range(qtestsize):
        qembed = query_test_set_index_embed[i]
        nums, similarity_ = [], []
        for j in range(dsize):
            cembed = cand_index_embed[j]
            nums.append((torch.dist(qembed, cembed, p=2), j))
        nums = heapq.nsmallest(choose, nums)
        for num in nums:
            query_test.append(query_test_set[i])
            cand.append(cand_set[num[1]])
    print('the first part', time.time()-start)
    start = time.time()
    env = Subtraj(cand, query_test)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/skip/your_model.h5')
    path_learn = r'./SoccerData/index_data/learn_'+str(pruning)+'/'
    similarity = run_evaluate_skip(choose, qtestsize, dsize)
    pickle.dump(similarity, open(path_learn+'similarity-learn', 'wb'), protocol=2)
    print('the second part', time.time()-start)
    
    tmp_c = []
    for c in cand:
        tmp_c.append(len(c))
    print(np.mean(tmp_c))
    
    tmp_c = []
    for c in cand_set:
        tmp_c.append(len(c))
    print(np.mean(tmp_c))
    
    tmp_c = []
    for c in query_test:
        tmp_c.append(len(c))
    print(np.mean(tmp_c))
    print('time cost of rls-skip learn index', pruning, time.time()-start)

##EFFECTIVENESS
for pruning in [0.5, 0.6, 0.7, 0.8]:
    path_learn = r'./SoccerData/index_data/learn_'+str(pruning)+'/'
    path_ground = r'./SoccerData/index_data/ground_'+str(dsize)+'_'+str(None)+'/'
    similarity = pickle.load(open(path_learn+'similarity-learn', 'rb'), encoding='bytes')
    res_ap, res_mr, res_rr = [], [], []
    for q in range(qtestsize):
        SUBsim = pickle.load(open(path_ground+'contain_'+str(q), 'rb'), encoding='bytes')
        t = SUBsim.index(min(similarity[q])) + 1
        res_ap.append(min(similarity[q])/SUBsim[0])
        res_mr.append(t)
        res_rr.append(t/len(SUBsim))
    print('effectiveness of learn idnex', np.mean(res_ap), np.mean(res_mr), np.mean(res_rr))
