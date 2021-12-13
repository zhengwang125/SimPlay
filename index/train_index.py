# -*- utf-8 -*-

import os
import pickle
import numpy as np
import heapq
import argparse
from  torch import nn
import torch
from utils import play2vec
import tensorflow as tf
import random

random.seed(0)    
tf.set_random_seed(0)
tf.reset_default_graph()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

path = r'./SoccerData/'
path1 = r'./SoccerData/index_data/'
m0 = play2vec(path)

pruning, dsize = 0.8, 500
    
class NeuNet(nn.Module):
    def __init__(self, args):
        super(NeuNet, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.in_features = args.embed_dim_input
        self.out_features = args.embed_dim_output
        
        self.fc1 = nn.Linear(self.in_features, self.out_features)
        self.drop = nn.Dropout(args.p_dropout)

    def forward(self, embeds, mode='train'):
        if mode == 'train':
            logit = self.fc1(embeds)
            logit = self.drop(logit)
        else:
            logit = self.fc1(embeds)
        return logit

def make_batches(samples, batch_size):
    num_batches = len(samples) // batch_size
    # working on only only full batches
    input_samples = samples[:num_batches * batch_size]
    for ind in range(0, len(input_samples), batch_size):
        input_batch = input_samples[ind:ind + batch_size]
        yield input_batch

def save(model, save_dir, save_prefix, steps, hr):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}_{}.pt'.format(save_prefix, steps,hr)
    torch.save(model.state_dict(), save_path)

def par():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early_stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save_interval', type=int, default=5,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save_path', type=str, default=path1+'model/', help='where to save the snapshot')
    parser.add_argument('-save_best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('--embed_dim_input', type=int, default=50,
                        help='input dimensions')
    parser.add_argument('--embed_dim_output', type=int, default=128,
                        help='output dimensions')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size for each minibatch')
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight_decay rate')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument("-p_dropout", type=float, default=0.3, help="The dropout probability")
    args = parser.parse_args()
    return args

def evaluate(NN, query_test_set_embed, cand_embed):
    path_ground = r'./SoccerData/index_data/ground_'+str(dsize)+'_'+str(None)+'/'
    choose = round(dsize*(1-pruning))
    query_test_set_index_embed = NN(torch.tensor(query_test_set_embed), 'test')
    cand_index_embed = NN(torch.tensor(cand_embed), 'test')
    res_ap = []
    for i in range(20):
        qembed = query_test_set_index_embed[i]
        nums, similarity_ = [], []
        for j in range(dsize):
            cembed = cand_index_embed[j]
            nums.append((torch.dist(qembed, cembed, p=2), j))
        nums = heapq.nsmallest(choose, nums)
        for num in nums:
            similarity_.append(similarity[i][num[1]])
        SUBsim = pickle.load(open(path_ground+'contain_'+str(i), 'rb'), encoding='bytes')
        res_ap.append(min(similarity_)/SUBsim[0])
    _ap = np.mean(res_ap)
    print('effectiveness with ap', _ap)
    return _ap

def get_item(query_train_set_embed, cand_embed, numpairs=300):
    item = [[],[],[]]
    for i in range(qtrainsize):
        for nums in range(int(numpairs/qtrainsize)):
            [a_idx, b_idx] = random.sample(range(500),2)
            a_val, b_val = similarity_train[i][a_idx], similarity_train[i][b_idx]
            if a_val < b_val:
                pos_idx, neg_idx = a_idx, b_idx
            else:
                pos_idx, neg_idx = b_idx, a_idx
            item[0].append(query_train_set_embed[i])
            item[1].append(cand_embed[pos_idx])
            item[2].append(cand_embed[neg_idx])
    item[0] = np.array(item[0])
    item[1] = np.array(item[1])
    item[2] = np.array(item[2])
    return item

def main(query_train_set_embed, query_test_set_embed, cand_embed):
    args = par()

    NN = NeuNet(args)
    
    #NN.load_state_dict(torch.load(args.save_path+str(best_ap)+'.pth'))
    
    optimizer = torch.optim.Adam(NN.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    best_ap = np.inf
    for epoch in range(100):
        
        for step in range(5):
            
            item = get_item(query_train_set_embed, cand_embed)
            input_batch_normal = next(make_batches(item[0], args.batch_size))

            input_batch_positive = next(make_batches(item[1], args.batch_size))

            input_batch_negative = next(make_batches(item[2], args.batch_size))            
            
            #normal
            logit_query = NN(torch.tensor(input_batch_normal))
            #positive
            logit_positive = NN(torch.tensor(input_batch_positive))
            # negative
            logit_negative = NN(torch.tensor(input_batch_negative))
            
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            output = triplet_loss(torch.DoubleTensor(logit_query.double()), torch.DoubleTensor(logit_positive.double()), torch.DoubleTensor(logit_negative.double()))
            print("---------------epoch {}: loss {}---------------".format(epoch, output))
            output.backward()
            optimizer.step()
            _ap = evaluate(NN, query_test_set_embed, cand_embed)
            if _ap < best_ap:
                best_ap = _ap
                torch.save(NN.state_dict(), args.save_path+str(best_ap)+'.pth')

def prepare():
    qtrainsize, qtestsize, dsize = 100, 100, 500
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    traj_tokens=sorted(traj_tokens,key=lambda i:len(i))[-4*dsize-qtrainsize-qtestsize:]
    random.shuffle(traj_tokens)
    return qtrainsize, qtestsize, dsize, traj_tokens

def obtain_embeds(traj_set):
    tmp = []
    for ts in traj_set:
        Q_outputs, Q_state, _ = m0.submit(ts)
        tmp.append(Q_state.reshape(-1))
    return np.array(tmp)
    
if __name__ == '__main__':
    qtrainsize, qtestsize, dsize, traj_tokens = prepare()
    # FOR TRAINING DATA PREPARE
    query_train_set = traj_tokens[0:qtrainsize]
    query_test_set = traj_tokens[qtrainsize:qtrainsize+qtestsize]
    cand_set = traj_tokens[-dsize:]
    cand = []
    for didx in range(dsize):
        cand.append(cand_set[didx])
    
    path_without = r'./SoccerData/index_data/without_'+str(dsize)+'_'+str(None)+'/'
    similarity_train=pickle.load(open(path1+'similarity-train_100_500', 'rb'), encoding='bytes')
    
    similarity=pickle.load(open(path_without+'similarity-without', 'rb'), encoding='bytes')
    
    query_train_set_embed = obtain_embeds(query_train_set)
    query_test_set_embed = obtain_embeds(query_test_set)
    cand_embed = obtain_embeds(cand)
    
    main(query_train_set_embed, query_test_set_embed, cand_embed)
