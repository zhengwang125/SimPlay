from rl_env import Subtraj
from rl_nn import DeepQNetwork
import numpy as np
import tensorflow as tf
from time import time
from exact import exact
import pickle

np.random.seed(1)
tf.set_random_seed(1)

def run_evaluate(elist):
    eva = []
    action_collect = []
    for e in elist:
        observation, steps = env.reset(e)
        for index in range(1, steps):
            action = RL.online_act(observation)
            action_collect.append(action)
            observation_, _= env.step(e, action, index)
            observation = observation_
        res = env.output()
        if SUBSIM[e] != 0:
            eva.append(res[0]/SUBSIM[e])
    aver_cr = sum(eva)/len(eva)
    print('average competive ratio:', aver_cr, sum(action_collect))
    return aver_cr

def run_subt():
    batch_size = 32
    check = 999999
    REWARD_CL = []
    TR_CR = []
    
    for episode in range(24200):
        observation, steps = env.reset(episode, 'T')

        REWARD = 0.0
        for index in range(1, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            # RL choose action based on observation
            action = RL.act(observation)

            # RL take action and get next observation and reward
            observation_, reward = env.step(episode, action, index, 'T')
            
            if reward != 0:                
                REWARD = REWARD + reward

            RL.remember(observation, action, reward, observation_, done)

            if done:
                RL.update_target_model()

                break
            if len(RL.memory) > batch_size:
                RL.replay(batch_size)

            # swap observation
            observation = observation_
        
        REWARD_CL.append(REWARD)
        # check states
        if SUBSIM[episode] != 0:
            TR_CR.append(env.output()[0]/SUBSIM[episode])

        if episode % 100 == 0:

             aver_cr = run_evaluate([i for i in range(24200, 24500)])
             if aver_cr < 1:
                 continue
             print('Training CR: {}, Validation CR: {}'.format(sum(TR_CR[-100:])/len(TR_CR[-100:]), aver_cr))
             if aver_cr < check or episode % 1000==0:
                 RL.save('./model_collect/test/sub-RL-' + str(aver_cr) + '.h5')
                 print('Save model at episode {} with competive ratio {}'.format(episode, aver_cr))
                 check = aver_cr
                
if __name__ == "__main__":
    # building subtrajectory env
    path1 = r'./SoccerData/subt_data/'
    env = Subtraj(path1+'cand_train', path1+'query_train')
    RL = DeepQNetwork(env.n_features, env.n_actions)
    #RL.load("./model_collect/test/your_model.h5")
    '''
    start = time()
    SUBSIM = []
    for i in range(0, 24500):#
        if i % 1000 == 0:
            print('process', i)
        __subsim, __subtraj, __subset = exact(env.cand_train_data[i], env.query_train_data[i])
        SUBSIM.append(__subsim)
    stop = time()
    print('exact length:', len(SUBSIM), 'time cost:',str(stop - start)+' seconds')
    pickle.dump(SUBSIM, open(path1+'SUBSIM', 'wb'), protocol=2)
    '''
    SUBSIM = pickle.load(open(path1+'SUBSIM', 'rb'), encoding='bytes')
    run_subt()