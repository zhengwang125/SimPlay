from exact import exact
import pickle
from time import time
from rl_nn import DeepQNetwork
from rl_env import Subtraj
#from rl_env_skip import Subtraj
import tensorflow as tf
import numpy as np

path = r'./SoccerData/'

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
            relative_rank.append(t*1.0/len(SUBSIM_RANK[e]))
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
            relative_rank.append(t*1.0/len(SUBSIM_RANK[e]))
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
    SUBSIM_SUBSET = []
    
    for i in elist:
        if i % 1000 == 0:
            print('process', i, time()-start)
        subsim, subtraj, subset = exact(cand_test_data[i], query_test_data[i])
        SUBSIM_TEST.append(subsim)
        subsort = sorted(subset.items(), key=lambda d: d[1])
        SUBSIM_RANK.append([j[0] for j in subsort])
        SUBSIM_SUBSET.append(subset)
    
    print('time cost:',str(time() - start)+' seconds')
    
    pickle.dump(SUBSIM_TEST, open(path1 + 'SUBSIM_TEST', 'wb'), protocol=2)
    pickle.dump(SUBSIM_RANK, open(path1 + 'SUBSIM_RANK', 'wb'), protocol=2)
    pickle.dump(SUBSIM_SUBSET, open(path1 + 'SUBSIM_SUBSET', 'wb'), protocol=2)
    
    SUBSIM_TEST = pickle.load(open(path1 + 'SUBSIM_TEST', 'rb'), encoding='bytes')
    SUBSIM_RANK = pickle.load(open(path1 + 'SUBSIM_RANK', 'rb'), encoding='bytes')
    SUBSIM_SUBSET = pickle.load(open(path1 + 'SUBSIM_SUBSET', 'rb'), encoding='bytes')
    #####################################################################################
    path1 = r'./SoccerData/simplification/Bottom-Up-0.75/subt_data/'
    
    cand_test = pickle.load(open(path1 + 'cand_test', 'rb'), encoding='bytes')
    query_test = pickle.load(open(path1 + 'query_test', 'rb'), encoding='bytes')
    cand_test_index = pickle.load(open(path1 + 'cand_test_index', 'rb'), encoding='bytes')
    query_test_index = pickle.load(open(path1 + 'query_test_index', 'rb'), encoding='bytes')
    Sourceint_Mapback = pickle.load(open(path1 + 'Sourceint_Mapback', 'rb'), encoding='bytes')

    env = Subtraj(path1+'cand_test', path1+'query_test', Sourceint_Mapback, cand_test_index, SUBSIM_SUBSET)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/test/your_model.h5')
    run_evaluate(elist)

    '''
    #skip testing
    env = Subtraj(path1+'cand_test', path1+'query_test', Sourceint_Mapback, cand_test_index, SUBSIM_SUBSET)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    RL.load('./model_collect/skip/your_model.h5')
    run_evaluate_skip(elist)
    #run_evaluate_skip_for_effi(elist)
    '''