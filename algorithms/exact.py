from utils import play2vec, pop_random
import random
import pickle
import numpy as np
from time import time
import tensorflow as tf

tf.reset_default_graph()
path = r'./SoccerData/'

random.seed (0)

m0 = play2vec(path)

def exact(traj_c, traj_q):
    Q_outputs, Q_state, _ = m0.submit(traj_q)
    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    subset = {}
    for i in range(len(traj_c)):
         C_outputs, C_state, _ = m0.submit(traj_c[i:])
         for j in range(C_outputs.shape[1]):
             #print('sub-range:', [i, i+j])
             temp = np.linalg.norm(Q_state - C_outputs[0,j,:])
             subset[(i, i+j)] = temp
             if temp < subsim:
                 subsim = temp
                 subtraj = [i, i+j]
    return subsim, subtraj, subset
    
if __name__ == '__main__':
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    start = time()
    Q_outputs, Q_state, _ = m0.submit(traj_tokens[0][0:2])
    print('1', Q_outputs, Q_state)
    Q_outputs, Q_state, init_state = m0.submit(traj_tokens[0][0])
    print('2', Q_outputs, Q_state)
    Q_outputs, Q_state, _ = m0.submit(traj_tokens[0][1],init_state)
    print('3', Q_outputs, Q_state)
#    (cand, query) = pop_random()
#    if len(traj_tokens[query]) > len(traj_tokens[cand]):
#        cand, query = query, cand
#    subsim, subtraj, subset = exact(traj_tokens[cand], traj_tokens[query])
#    print('time cost:',str(time() - start)+' seconds')
#    print('query:', traj_tokens[query])
#    print('sub-trajectory Interval', subtraj)
#    print('sub-similarity', subsim)
#    print('sub-trajectory is', traj_tokens[cand][subtraj[0]:subtraj[1] + 1])
#    subsort = sorted(subset.items(), key=lambda d: d[1])