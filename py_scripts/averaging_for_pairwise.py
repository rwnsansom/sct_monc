#!/usr/bin/env python
import numpy as np
from itertools import combinations

def average_through_dimensions(colour, data_type, ext):
    reshaped_array = np.reshape(colour, (10,10,10,10,10,10))
    for count, (i, j) in enumerate(combinations(range(6), 2)):

        listed = [0,1,2,3,4,5]
        if i==0 or j==0:
            listed.remove(5)
        if i==1 or j==1:
            listed.remove(4)
        if i==2 or j==2:
            listed.remove(3)
        if i==3 or j==3:
            listed.remove(2)
        if i==4 or j==4:
            listed.remove(1)
        if i==5 or j==5:
            listed.remove(0)
        print(listed)
        print(np.shape(reshaped_array))

        mean = reshaped_array.mean(axis=tuple(listed))
        np.savetxt(f"../predictions/mean_pairs/mean_pair_{data_type}_{i}{j}_nv_sc_beginning{ext}.csv", mean, delimiter=',')

def slice_through_and_average(colour, index_1, index_2, index_3, data_type, ext):
    reshaped_array = np.reshape(colour, (10,10,10,10,10,10))
    listed = [0,1,2,3,4,5]
    if index_1==0 or index_2==0 or index_3==0:
        listed.remove(5)
    if index_1==1 or index_2==1 or index_3==1:
        listed.remove(4)
    if index_1==2 or index_2==2 or index_3==2:
        listed.remove(3)
    if index_1==3 or index_2==3 or index_3==3:
        listed.remove(2)
    if index_1==4 or index_2==4 or index_3==4:
        listed.remove(1)
    if index_1==5 or index_2==5 or index_3==5:
        listed.remove(0)

    mean_3d = reshaped_array.mean(axis=tuple(listed))
    for i in range(len(mean_3d)):
        slice_i = mean_3d[:,i,:]
        np.savetxt(f"../predictions/mean_slices/mean_slice_{data_type}_{index_1}{index_2}{index_3}_slice{i}_nv_sc_beginning{ext}.csv", slice_i, delimiter=',')

data_type = "transition_time"   # transition_time or rwp_mean
noise = ""
ext = ""

design = np.loadtxt(f"../predictions/grid10_design{ext}.csv", delimiter=',', skiprows=1)
pred = np.loadtxt(f"../predictions/grid10_{data_type}_nv_sc_beginning{ext}.csv", delimiter=',', skiprows=1)

average_through_dimensions(pred[:,1], data_type, ext)
slice_through_and_average(pred[:,1], 2, 5, 4, data_type, ext)