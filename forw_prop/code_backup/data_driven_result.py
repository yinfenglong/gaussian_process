#!/usr/bin/env python
# coding=utf-8
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def datas_get( csv_name ):
    data = pd.read_csv( csv_name,index_col=True) #othrewise dt will be regarded as index
    # u_k = np.array( data.iloc[:, 1:5] )
    # x_k = np.array( data.iloc[:, 5:15] )
    # x_k_1_mpc = np.array( data.iloc[:, 15:] )
    state_in = np.array( data.iloc[:,0])
    error = np.array( data.iloc[:,2])

def show_result(i):
    # show results of gazebo and rk4
    f,ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot( state_in[:, i], state_in[:,i],'b*')
    plt.show()

if __name__ == '__main__':
	state_in = datas_get(sys.argv[1])
	vx = 
