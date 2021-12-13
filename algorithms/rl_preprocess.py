from utils import pop_random
import pickle
from sklearn.model_selection import train_test_split
import random

random.seed(0)
path = r'./SoccerData/'

if __name__ == '__main__':
    traj_tokens=pickle.load(open(path+'source_int', 'rb'), encoding='bytes')
    Cand = []
    Query = []
    
    for i in range(35000):
        (cand, query) = pop_random()
        if len(traj_tokens[query]) > len(traj_tokens[cand]):
            cand, query = query, cand
        Cand.append(traj_tokens[cand])
        Query.append(traj_tokens[query])

    cand_train, cand_test, query_train, query_test = train_test_split(Cand, Query, random_state=1, train_size=0.7)

    pickle.dump(cand_train, open(path + 'subt_data/cand_train', 'wb'), protocol=2)
    pickle.dump(cand_test, open(path + 'subt_data/cand_test', 'wb'), protocol=2)
    pickle.dump(query_train, open(path + 'subt_data/query_train', 'wb'), protocol=2)
    pickle.dump(query_test, open(path + 'subt_data/query_test', 'wb'), protocol=2)
    