from rl_env_skip import Subtraj
from rl_nn import DeepQNetwork
import numpy as np
import tensorflow as tf
from time import time
import pickle

path = r'./SoccerData/'

np.random.seed(1)
tf.set_random_seed(1)

def run_evaluate(elist):
    eva = []
    skip_f = False
    split_f = False
    for e in elist:
        observation, steps, INX = env.reset(e, 'E')
        for index in range(1, steps):
            if index <= INX:
                continue
            action = RL.online_act(observation)
            if action == 1:
                split_f = True
            if action > 1:
                skip_f = True
            observation_, _, done, INX = env.step(e, action, index, 'E')
            observation = observation_
        res = env.output(index, e)
        if SUBSIM[e] != 0:
            eva.append(res[0]/SUBSIM[e])
    aver_cr = sum(eva)/len(eva)
    print('average competive ratio:', aver_cr, skip_f, split_f)
    return aver_cr

def run_subt():
    batch_size = 32
    check = 999999
    REWARD_CL = []
    TR_CR = []

    for episode in range(24200):
        observation, steps, INX = env.reset(episode, 'T')
        REWARD = 0.0
        for index in range(1, steps):
            if index <= INX:
                continue
            # RL choose action based on observation
            action = RL.act(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, INX = env.step(episode, action, index, 'T')#E
            
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
            TR_CR.append(env.output(index, episode, '')[0]/SUBSIM[episode])
        if episode % 100 == 0:
             aver_cr = run_evaluate([i for i in range(24200, 24500)])
             if aver_cr < 1:
                 continue
             print('Training CR: {}, Validation CR: {}'.format(sum(TR_CR[-100:])/len(TR_CR[-100:]), aver_cr))
             if aver_cr < check or episode % 1000 == 0:
                 RL.save('./model_collect/skip/sub-RL-' + str(aver_cr) + '.h5')
                 print('Save model at episode {} with competive ratio {}'.format(episode, aver_cr))
                 check = aver_cr
                
if __name__ == "__main__":
    # building subtrajectory env
    path1 = r'./SoccerData/subt_data/'
    env = Subtraj(path1+'cand_train', path1+'query_train')
    RL = DeepQNetwork(env.n_features, env.n_actions)
    #RL.load("./model_collect/skip/your_model.h5")
    SUBSIM = pickle.load(open(path1+'SUBSIM', 'rb'), encoding='bytes')
    run_subt()