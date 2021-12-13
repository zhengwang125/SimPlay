# -*- coding: utf-8 -*-
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from utils import corrupt_drop

def get_ogm_index(division, delta, xlim, ylim, drop_id):
    '''
    division: small segmentation
    delta: grid length
    xlim and ylim: boundary of soccer field
    '''
    traj_index = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44]
    temp = []
    for row in division:
        if sum(row)==-46.0:
            continue
        for i in traj_index:
            x = row[i]
            y = row[i+1]
            x_grid = int((x-xlim[0])/delta)
            y_grid = int((y-ylim[0])/delta)
            temp.append((x_grid,y_grid))
    return temp

def handle_data_error(grid_index, xlim, ylim, delta):
    '''
    grid_index: index for small segmentation
    xlim and ylim: boundary of soccer field
    delta: grid length
    '''
    res_grid_index = []
    #handle data error
    lencol = math.ceil((xlim[1]-xlim[0])/delta)
    lenrow = math.ceil((ylim[1]-ylim[0])/delta)
    for pair in grid_index:
        if pair[1] >= lenrow:
            xcoor = lenrow-1
        elif pair[1] < 0:
            xcoor = 0
        else:
            xcoor = pair[1]
        
        if pair[0] >= lencol:
            ycoor = lencol-1
        elif pair[0] < 0:
            ycoor = 0
        else:
            ycoor = pair[0]
        res_grid_index.append((ycoor,xcoor))
    return res_grid_index
    
def grid_index_to_array(grid_index, xlim, ylim, delta):
    '''
    grid_index: index for small segmentation
    xlim and ylim: boundary of soccer field
    delta: grid length
    '''
    lencol = math.ceil((xlim[1]-xlim[0])/delta)
    lenrow = math.ceil((ylim[1]-ylim[0])/delta)
    matrix = [[0 for i in range(lencol)] for j in range(lenrow)]
    for pair in grid_index:
        matrix[pair[1]][pair[0]]=1
    return np.array(matrix)

def viz_ogm(matrix):
    ones = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
    plt.imshow(ones - matrix, cmap='gray')
    plt.show()

def mseq2ogm(seq=b'sequence_1', data=None, segment=10, delta=1.0, xlim=[-52.5,52.5], ylim=[-34,34], drop_id=[]):        
    '''
    seq: game clips
    data: datasets
    segment: video division length
    delta: grid length
    xlim and ylim: boundary of soccer field
    '''
    for id in drop_id:
        data[seq][id]=[-1.0]*46
    Division = np.array_split(data[seq], round(len(data[seq])/segment), axis = 0)
    GRID_INDEX = []
    for division in Division:
        grid_index = get_ogm_index(division, delta, xlim, ylim, drop_id)
        grid_index = handle_data_error(grid_index, xlim, ylim, delta)
        grid_index = list(set(grid_index))
        GRID_INDEX.append(grid_index)
    
    return GRID_INDEX

if __name__ == '__main__':
    print('occupancy grid maps')
    path1=r'./SoccerData/'
    path2=r'./SoccerData/simplification/Bottom-Up-0.75/' #Top-Down-0.75, Uniform-0.75
    
    train_data=pickle.load(open(path2+'simplified_train_data', 'rb'), encoding='bytes')
    drop_id = corrupt_drop(train_data, rate_drop = 0.2) #rate_drop from 0.2 to 0.6
    xlim = [-52.5,52.5]
    ylim = [-34,34]
    segment = 10
    delta = 3.0
    GRID_INDEX = mseq2ogm(seq=b'sequence_18', data=train_data, segment=segment, delta=delta, xlim=xlim, ylim=ylim, drop_id=drop_id[b'sequence_18'])
    matrix = grid_index_to_array(GRID_INDEX[1], xlim, ylim, delta)
    viz_ogm(matrix)
    drop_ogm_train_data = []
    counter = 1
    for key in train_data.keys():
        #print(key)
        if counter % 500 == 0:
            print('processing:',counter,'/',len(train_data))
        drop_ogm_train_data.append(mseq2ogm(seq=key, data=train_data, segment=segment, delta=delta, xlim=xlim, ylim=ylim, drop_id=drop_id[key]))
        counter = counter + 1
    pickle.dump(drop_ogm_train_data, open(path2+'drop_ogm_train_data', 'wb'), protocol=2)