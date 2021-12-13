from heuristic import heuristic, adjust
from exact import exact
import pickle
from time import time
from rl_main import Subtraj, DeepQNetwork
#from rl_main_skip import Subtraj, DeepQNetwork
from fixed_length import fixed
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)
delayK = 1

def run_evaluate(elist):
    eva = []
    rank = []
    relative_rank = []
    start = time()
    for e in elist:
        observation, steps = env.reset(e, 'E')
        total = (steps) * (steps + 1) / 2
        for index in range(1, steps):
            action = RL.fast_online_act(observation)
            observation_, _ = env.step(e, action, index, 'E')
            observation = observation_
        res = env.output()   
        if SUBSIM_TEST[e] != 0:
            eva.append(res[0]/SUBSIM_TEST[e])
        elif res[0] == SUBSIM_TEST[e]:
            eva.append(1.0)
        if tuple(res[1]) in SUBSIM_RANK[e]:
            t = SUBSIM_RANK[e].index(tuple(res[1])) + 1
            rank.append(t)
            relative_rank.append(t*1.0/total)
        else:
            print((tuple(res[1]), res[0]))

    print('10000 time cost:',str(time() - start)+' seconds')
    print('average competive ratio:', sum(eva)/len(eva))
    print('mean rank:', sum(rank)/len(rank))
    print('aver_relative_rank:', sum(relative_rank)/len(relative_rank))
            
def run_evaluate_skip(elist):
    total_pts = 0
    skip_pts = 0
    eva = []
    rank = []
    relative_rank = []
    start = time()
    for e in elist:
        observation, steps, INX = env.reset(e, 'E')
        total_pts = total_pts + steps
        total = (steps) * (steps + 1) / 2
        for index in range(1, steps):
            if index <= INX:
                skip_pts = skip_pts + 1
                continue
            action = RL.fast_online_act(observation)
            observation_, _, done, INX  = env.step(e, action, index, 'E')
            observation = observation_
        res = env.output(index, e)
        if SUBSIM_TEST[e] != 0:
            eva.append(res[0]/SUBSIM_TEST[e])
        elif res[0] == SUBSIM_TEST[e]:
            eva.append(1.0)
        if tuple(res[1]) in SUBSIM_RANK[e]:
            t = SUBSIM_RANK[e].index(tuple(res[1])) + 1
            rank.append(t)
            relative_rank.append(t*1.0/total)
        else:
            print((tuple(res[1]), res[0]))

    print('10000 time cost:',str(time() - start)+' seconds')
    print('average competive ratio:', sum(eva)/len(eva))
    print('mean rank:', sum(rank)/len(rank))
    print('aver_relative_rank:', sum(relative_rank)/len(relative_rank))
    print(total_pts, skip_pts)
    
def run_evaluate_skip_for_effi(elist):
    start = time()
    for e in elist:
        observation, steps, INX = env.reset(e, 'E')
        for index in range(1, steps):
            if index <= INX:
                continue
            action = RL.fast_online_act(observation)
            observation_, _, done, INX  = env.step(e, action, index, 'E')
            observation = observation_
    print('10000 time cost:',str(time() - start)+' seconds')
    
if __name__ == '__main__':
    path1 = r'./SoccerData/subt_data/'
    cand_test_data = pickle.load(open(path1+'cand_test', 'rb'), encoding='bytes')
    query_test_data = pickle.load(open(path1+'query_test', 'rb'), encoding='bytes')
    length =len(cand_test_data)
    elist = [i for i in range(10000)]
    
    start = time()
    SUBSIM_TEST = []
    SUBSIM_RANK = []
    for i in elist:
        if i % 1000 == 0:
            print('process', i, time()-start)
        subsim, subtraj, subset = exact(cand_test_data[i], query_test_data[i])
        SUBSIM_TEST.append(subsim)
        subsort = sorted(subset.items(), key=lambda d: d[1])
        SUBSIM_RANK.append([j[0] for j in subsort])
    
    print('time cost:',str(time() - start)+' seconds')
    
    pickle.dump(SUBSIM_TEST, open(path1 + 'SUBSIM_TEST', 'wb'), protocol=2)
    pickle.dump(SUBSIM_RANK, open(path1 + 'SUBSIM_RANK', 'wb'), protocol=2)
    
    SUBSIM_TEST = pickle.load(open(path1 + 'SUBSIM_TEST', 'rb'), encoding='bytes')
    SUBSIM_RANK = pickle.load(open(path1 + 'SUBSIM_RANK', 'rb'), encoding='bytes')
    
    '''
    POS: maintain prefix only O(n)
    POS-D: maintain prefix and delay k steps only O(n)
    PSS: maintain prefix and backward-suffix O(n)
    '''

    for oo in ['POS','PSS','POS-D#5','fix#length#5']:
        print('algorithm:', oo)
        oot = oo.split('#')
        opt = oot[0]
        if len(oot) == 2:
            delayK = int(oot[1])
        if len(oot) == 3:
            par = int(oot[2])
        eva = []
        rank = []
        relative_rank = []
        start = time()
        for i in elist:
            #print('steps:', len(cand_test_data[i]), len(SUBSIM_RANK[i]))
            total = len(cand_test_data[i]) * (len(cand_test_data[i]) + 1) / 2
            if opt == 'fix':
                ap_subsim, ap_subtraj = fixed(cand_test_data[i], query_test_data[i], par)
            else:
                ap_subsim, ap_subtraj = heuristic(cand_test_data[i], query_test_data[i], opt, delayK)
                ap_subsim, ap_subtraj = adjust(ap_subsim, ap_subtraj, cand_test_data[i], query_test_data[i], opt)
            if SUBSIM_TEST[i] != 0:
                eva.append(ap_subsim/SUBSIM_TEST[i])
            elif ap_subsim == SUBSIM_TEST[i]:
                eva.append(1.0)
            if tuple(ap_subtraj) in SUBSIM_RANK[i]:
                t = SUBSIM_RANK[i].index(tuple(ap_subtraj)) + 1
                rank.append(t)
                relative_rank.append(t*1.0/total)
            
        print('10000 time cost:',str(time() - start)+' seconds')
        print('average competive ratio:', sum(eva)/len(eva))
        print('mean rank:', sum(rank)/len(rank))
        print('relative rank:', sum(relative_rank)/len(relative_rank))


    env = Subtraj(path1+'cand_test', path1+'query_test')
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/test/your_model.h5')
    run_evaluate(elist)

    '''
    #skip testing
    env = Subtraj(path1+'cand_test', path1+'query_test')
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/skip/your_model.h5')
    run_evaluate_skip(elist)
    #run_evaluate_skip_for_effi(elist)    
    '''