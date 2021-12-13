# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:45:29 2021

@author: wang_zheng
"""
import pickle

path1=r'./SoccerData/'
path2=r'./SoccerData/simplification/Bottom-Up-0.75/' #Top-Down-0.75, Uniform-0.75
segment = 10

source_int_original = pickle.load(open(path1+'source_int', 'rb'), encoding='bytes')
source_int_simplified = pickle.load(open(path2+'source_int', 'rb'), encoding='bytes')
simplified_train_index = pickle.load(open(path2+'simplified_train_index', 'rb'), encoding='bytes')

c = 0
Sourceint_Mapback = []
for key in simplified_train_index:
    sourceint_mapback = []
    for i in range(len(source_int_simplified[c])):
        if i == 0:
            sourceint_mapback.append(0)
        elif i == len(source_int_simplified[c]) - 1:
            sourceint_mapback.append(len(source_int_original[c])-1)
        else:
            sourceint_mapback.append(int(simplified_train_index[key][i*segment]/10))
    Sourceint_Mapback.append(sourceint_mapback)
    if c%100==0:
        print('current', c, 'map back', sourceint_mapback, [j for j in range(len(source_int_original[c]))])
    c+=1
pickle.dump(Sourceint_Mapback, open(path2+'Sourceint_Mapback', 'wb'), protocol=2)
    
        