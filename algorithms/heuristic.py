from utils import play2vec, pop_random
import random
import pickle
import numpy as np
from time import time
from exact import exact
path = r'./SoccerData/'
random.seed (0)
import tensorflow as tf
tf.reset_default_graph()

m0 = play2vec(path)

def heuristic_suffix_opt(invQ, Q, traj_c, index, opt, each_step_b_c):
    if traj_c[index:] == []:
        return 999999
    if opt == 'POS' or opt == 'POS-D':
        return 999999
    if opt == 'PSS':
        return np.linalg.norm(invQ - each_step_b_c[0, len(traj_c)-index-1, :])

def heuristic(traj_c, traj_q, opt, delay_K = 1):
    delay = 0
    each_step_f_q, _, _ = m0.submit(traj_q)
    Q = each_step_f_q[0,-1,:]
    invQ = -1
    each_step_b_c = -1
    if opt == 'PSS':
        each_step_b_q, _, _ = m0.submit(traj_q[::-1])
        invQ = each_step_b_q[0,-1,:]
        each_step_b_c, _, _ = m0.submit(traj_c[::-1])
    
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    split_point = 0
    init_state = None
    pos_d_coll = []
    pos_d_f = False
    temp = 'non'
    if opt != 'POS-D':
        for i in range(len(traj_c)):
            #submit prefix
            _, h, init_state = m0.submit(traj_c[i:i+1], init_state)
            presim = np.linalg.norm(Q - h[0,:])
            sufsim = heuristic_suffix_opt(invQ, Q, traj_c, i+1, opt, each_step_b_c)
            #print('-> maintain:', subtraj, subsim)
            #print('prefix:', [split_point, i], presim)
            #print('suffix:', [i+1, len(traj_c)-1], sufsim)
            if presim < subsim or sufsim < subsim:
                temp = i + 1
                subsim = min(presim, sufsim)
                if presim < sufsim:
                    subtraj = [split_point, (temp-1)]
                else:
                    subtraj = [temp, len(traj_c)-1]
                split_point = temp
                init_state = None
    else:
        i = -1
        while True:
            i = i + 1
            if temp != 'non':
                i = temp
                temp = 'non'
            if i == len(traj_c) - 1:
                break
            #submit prefix
            _, h, init_state = m0.submit(traj_c[i:i+1], init_state)
            presim = np.linalg.norm(Q - h[0,:])
            if pos_d_f == False and presim < subsim: #open delay
                pos_d_f = True
            if pos_d_f == True and delay < delay_K:
                delay = delay + 1
                pos_d_coll.append((presim, i))
                continue
            if pos_d_f == True and delay == delay_K:
                sort = sorted(pos_d_coll, key=lambda d: d[0])
                temp = sort[0][1] + 1
                subsim = sort[0][0]
                subtraj = [split_point, (temp-1)]
                split_point = temp
                init_state = None
                delay = 0
                pos_d_f = False
                pos_d_coll = []
        if subsim == 999999:
            sort = sorted(pos_d_coll, key=lambda d: d[0])
            temp = sort[0][1] + 1
            subsim = sort[0][0]
            subtraj = [split_point, (temp-1)]
            
    return subsim, subtraj

def adjust(ap_subsim, ap_subtraj, traj_c, traj_q, opt):
    if opt == 'PSS':
        if ap_subtraj[1] == len(traj_c) - 1:
            each_step_f_q, _, _ = m0.submit(traj_q)
            Q = each_step_f_q[0,-1,:]
            suffix_h, _, _ = m0.submit(traj_c[ap_subtraj[0]:])
            return np.linalg.norm(Q - suffix_h[-1,0,:]), ap_subtraj
    return ap_subsim, ap_subtraj

if __name__ == '__main__':
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    (cand, query) = pop_random()
    if len(traj_tokens[query]) > len(traj_tokens[cand]):
        cand, query = query, cand
    
    start = time()
    subsim, subtraj, subset = exact(traj_tokens[cand], traj_tokens[query])
    stop = time()
    print('time cost:',str(stop - start)+' seconds')
    print('candidate, query:', traj_tokens[cand], traj_tokens[query])
    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim)
    
#    '''
#    POS: maintain prefix only O(n)
#    POS-D: maintain prefix and delay k steps only O(n)
#    PSS: maintain prefix and backward-suffix O(n)
#    '''
    opt = 'POS-D'
    start = time()
    ap_subsim, ap_subtraj = heuristic(traj_tokens[cand], traj_tokens[query], opt, 1)
    stop = time()
    print('time cost:',str(stop - start)+' seconds')
    ap_subsim, ap_subtraj = adjust(ap_subsim, ap_subtraj, traj_tokens[cand], traj_tokens[query], opt)
    print('ap-sub-trajectory', ap_subtraj)
    print('ap-sub-similarity', ap_subsim)
    print('Competitive Ratio', ap_subsim/subsim)
    subsort = sorted(subset.items(), key=lambda d: d[1])
    