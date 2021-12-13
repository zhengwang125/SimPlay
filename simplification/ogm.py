# -*- coding: utf-8 -*-
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import heapq
from time import time
import heapq
from heapq import heappush, heappop, _siftdown, _siftup

def get_ogm_index(division, delta, xlim, ylim):
    '''
    division: small segmentation
    delta: grid length
    xlim and ylim: boundary of soccer field
    '''
    traj_index = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44]
    temp = []
    for row in division:
        for i in traj_index:
            x = row[i]
            y = row[i+1]
            x_grid = int((x-xlim[0])/delta)
            y_grid = int((y-ylim[0])/delta)
            #print('(x_grid, y_grid):', x_grid, y_grid)
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
        #print(pair)
        matrix[pair[1]][pair[0]]=1
    return np.array(matrix)

def viz_ogm(matrix):
    ones = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
    plt.imshow(ones - matrix, cmap='gray')
    plt.show()

def mseq2ogm(seq=b'sequence_1', data=None, segment=10, delta=1.0, xlim=[-52.5,52.5], ylim=[-34,34]):        
    '''
    seq: game clips
    data: datasets
    segment: video division length
    delta: grid length
    xlim and ylim: boundary of soccer field
    '''
    Division = np.array_split(data[seq], round(len(data[seq])/segment), axis = 0)
    #MATRIX = []
    GRID_INDEX = []
    for division in Division:
        grid_index = get_ogm_index(division, delta, xlim, ylim)
        grid_index = handle_data_error(grid_index, xlim, ylim, delta)
        grid_index = list(set(grid_index))
        GRID_INDEX.append(grid_index)
        #matrix = grid_index_to_array(grid_index, xlim, ylim, delta)
        #viz_ogm(matrix)
        #MATRIX.append(matrix)
    
    return GRID_INDEX

def ped_op_with_index(segment, start_index=0):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0, start_index
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            pm = segment[i]
            tmp_e = 0.0
            for j in range(0, 46, 2):            
                A = pe[j+1] - ps[j+1]
                B = ps[j] - pe[j]
                C = pe[j] * ps[j+1] - ps[j] * pe[j+1]
                if A == 0 and B == 0:
                    tmp_e += 0.0
                else:
                    tmp_e += abs((A * pm[j] + B * pm[j+1] + C)/ np.sqrt(A * A + B * B))
            if e < tmp_e:
                e = tmp_e
                returns = i
        return e, start_index+returns

def Top_Down(points, buffer_size):
    seg_err = {}
    err, idx = ped_op_with_index(points, 0) #ped_op sed_op speed_op dad_op
    seg_err[(0, len(points) - 1)] = (err, idx)
    simp = [0, len(points) - 1]
    while True:
        if len(simp) == buffer_size:
            break
        res = heapq.nlargest(1,seg_err.items(),lambda x:x[1][0])[0]
        key = res[0]
        select_index = res[1][1]
        del seg_err[key]
        select_start, select_end = key[0], key[1]
        #print(select_start, select_index, select_end)
        err, idx = ped_op_with_index(points[select_start:select_index+1], select_start)
        seg_err[(select_start, select_index)] = (err, idx)
        
        err, idx = ped_op_with_index(points[select_index:select_end+1], select_index)
        seg_err[(select_index, select_end)] = (err, idx)
        
        simp.append(select_index)

    return simp

def simplification_topdown(train_data, ratio):
    simplified_train_data = {}
    simplified_train_index = {}
    count_t = 0
    for key in train_data:
        if count_t%100 == 0:
            print('process ', count_t, '/', str(len(train_data)))
        count_t += 1
        points = train_data[key]
        buffer_size = int(ratio*len(points))
        if buffer_size < 2:
            simplified_train_data[key]=np.array([points[0], points[-1]])
            continue
        simp = Top_Down(points, buffer_size)
        simp.sort()
        sim_traj = [points[v] for v in simp]
        simplified_train_data[key] = np.array(sim_traj)
        simplified_train_index[key] = simp
    return simplified_train_data, simplified_train_index

def simplification_uniform(train_data, per, keep):
    simplified_train_data = {}
    simplified_train_index = {}
    count_t = 0
    for key in train_data:
        if count_t%100 == 0:
            print('process ', count_t, '/', str(len(train_data)))
        count_t += 1
        points = train_data[key]
        sim_traj = []
        sim_index = []
        for i in range(len(points)):
            if i == 0 or i == len(points)-1:
                sim_traj.append(points[i])
                sim_index.append(i)
                continue
            if i%per < keep:
                sim_traj.append(points[i])
                sim_index.append(i)
        simplified_train_data[key] = np.array(sim_traj)
        simplified_train_index[key] = sim_index
    return simplified_train_data, simplified_train_index

def read(p, points, heap, buffer, observation):
    buffer.append(p)
    observation.append(0.0)
    err, _ = ped_op_with_index(points[buffer[-3]:buffer[-1] + 1])
    observation[-2] = err
    heapq.heappush(heap, (observation[-2]))

def delete_heap(heap, nodeValue):
    leafValue = heap[-1]
    i = heap.index(nodeValue)
    if nodeValue == leafValue:
        heap.pop(-1)
    elif nodeValue <= leafValue: # similar to heappop
        heap[i], heap[-1] = heap[-1], heap[i]
        minimumValue = heap.pop(-1)
        if heap != []:
            _siftup(heap, i)
    else: # similar to heappush
        heap[i], heap[-1] = heap[-1], heap[i]
        minimumValue = heap.pop(-1)
        _siftdown(heap, 0, i)

def Bottom_Up(points, heap, buffer, observation):
    rem = observation[1:-1].index(heap[0]) + 1
    #rem = np.argmin(observation[1:-1]) + 1
    if rem > 1:
        delete_heap(heap, observation[rem - 1])
        err, _ = ped_op_with_index(points[buffer[rem - 2]:buffer[rem + 1] + 1])
        observation[rem - 1] = err
        heapq.heappush(heap, observation[rem - 1])
    if rem + 2 < len(buffer):
        delete_heap(heap, observation[rem + 1])
        err, _ = ped_op_with_index(points[buffer[rem - 1]:buffer[rem + 2] + 1]) 
        observation[rem + 1] = err
        heapq.heappush(heap, observation[rem + 1])
    delete_heap(heap, observation[rem])
    del buffer[rem]
    del observation[rem]

def simplification_bottomup(train_data, ratio):
    simplified_train_data = {}
    simplified_train_index = {}
    count_t = 0
    for key in train_data:
        if count_t%100 == 0:
            print('process ', count_t, '/', str(len(train_data)))
        count_t += 1
        points = train_data[key]
        buffer_size = int(ratio*len(points))
        if buffer_size < 3:
            simplified_train_data[key]=np.array([points[0], points[-1]])
            simplified_train_index[key]=[0, len(points)-1]
            continue
        steps = len(points)
        buffer = [0, 1]  #save traj index
        observation = [0.0, 0.0] #save errors
        
        heap = []
        for i in range(2, steps):
            read(i, points, heap, buffer, observation)
        while True:
            if len(buffer) == buffer_size:
                break
            Bottom_Up(points, heap, buffer, observation)
        simplified_train_data[key] = np.array([points[i] for i in buffer])
        simplified_train_index[key] = buffer
    return simplified_train_data, simplified_train_index


if __name__ == '__main__':
    print('occupancy grid maps')
    start = time()
    path1=r'./SoccerData/'
    path2=r'./SoccerData/simplification/Bottom-Up-0.75/' #Top-Down-0.75, Uniform-0.75
    train_data=pickle.load(open(path1+'train_data.pkl', 'rb'), encoding='bytes')
    
    '''
    train_data, train_index = simplification_topdown(train_data, ratio=0.25)
    print('simplification time cost:',str(time() - start)+' seconds')
    pickle.dump(train_data, open(path2+'simplified_train_data', 'wb'), protocol=2)
    pickle.dump(train_index, open(path2+'simplified_train_index', 'wb'), protocol=2)
    '''
    '''
    train_data, train_index = simplification_uniform(train_data, per=4, keep=1)
    print('simplification time cost:', str(time() - start)+' seconds')
    pickle.dump(train_data, open(path2+'simplified_train_data', 'wb'), protocol=2)
    pickle.dump(train_index, open(path2+'simplified_train_index', 'wb'), protocol=2)
    '''
    
    train_data, train_index = simplification_bottomup(train_data, ratio=0.75)
    print('simplification time cost:', str(time() - start)+' seconds')
    pickle.dump(train_data, open(path2+'simplified_train_data', 'wb'), protocol=2)
    pickle.dump(train_index, open(path2+'simplified_train_index', 'wb'), protocol=2)
    
    train_data=pickle.load(open(path2+'simplified_train_data', 'rb'), encoding='bytes')    
    xlim = [-52.5,52.5]
    ylim = [-34,34]
    segment = 10
    delta = 3.0
    GRID_INDEX = mseq2ogm(seq=b'sequence_18', data=train_data, segment=segment, delta=delta, xlim=xlim, ylim=ylim)
    matrix = grid_index_to_array(GRID_INDEX[1], xlim, ylim, delta)
    viz_ogm(matrix)
    ogm_train_data = []
    ogm_train_key = []
    counter = 1
    for key in train_data.keys():
        #print(key)
        if counter % 500 == 0:
            print('processing:',counter,'/',len(train_data))
        ogm_train_data.append(mseq2ogm(seq=key, data=train_data, segment=segment, delta=delta, xlim=xlim, ylim=ylim))
        ogm_train_key.append(key)
        counter = counter + 1
    pickle.dump(ogm_train_data, open(path2+'ogm_train_data', 'wb'), protocol=2)
    pickle.dump(ogm_train_key, open(path2+'ogm_train_key', 'wb'), protocol=2)