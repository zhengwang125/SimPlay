from utils import play2vec, pop_random
import random
import pickle
import numpy as np
from time import time
from exact import exact
import tensorflow as tf

tf.reset_default_graph()
path = r'./SoccerData/'

random.seed (0)

m0 = play2vec(path)

def fixed(traj_c, traj_q, par):    
    L = len(traj_q)
    L_lo = min(len(traj_c), int((L - par)))
    L_up = min(len(traj_c), int((L + par)))

    each_step_f_q, _, _ = m0.submit(traj_q)
    Q = each_step_f_q[0,-1,:]
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    for i in range(len(traj_c)):
         each_step_f_c, _, _  = m0.submit(traj_c[i:i+L_up])
         for j in range(each_step_f_c.shape[1]):

             if (j + 1) <  L_lo:
                 continue
             temp = np.linalg.norm(Q - each_step_f_c[0,j,:])
             if temp < subsim:
                 subsim = temp
                 subtraj = [i, i+j]
    return subsim, subtraj
    
if __name__ == '__main__':
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    (cand, query) = pop_random()
    if len(traj_tokens[query]) > len(traj_tokens[cand]):
        cand, query = query, cand
    start = time()
    par = 2
    ap_subsim, ap_subtraj = fixed(traj_tokens[cand], traj_tokens[query], par)
    subsim, subtraj, subset = exact(traj_tokens[cand], traj_tokens[query])
    print('time cost:',str(time() - start)+' seconds')
    print('candidate, query:', traj_tokens[cand], traj_tokens[query])
    print('sub-trajectory', subtraj)
    print('sub-similarity', subsim)
    print('ap-sub-trajectory', ap_subtraj)
    print('ap-sub-similarity', ap_subsim)
    print('Competitive Ratio', ap_subsim/subsim)