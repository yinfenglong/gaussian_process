#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-09-02
usage: 
        python3 compare_control_with_p_npy.py title_name
        python3 compare_control_with_p_npy.py exp_data_control_with_p.npy 20210913_2_circle_30s q330
'''

import numpy as np
from matplotlib import pyplot as plt
import sys
import os.path

def load_npy(np_file):
    exp_data = np.load(np_file, allow_pickle=True)
    data_length = exp_data.shape[0]
    print("Data length:", data_length)
    # got_data = False

    # t = np.arange(0., 0.01*data_length, 0.01)
    wx_p = []
    wy_p = []
    wz_p = []
    thrust_p = []

    wx = []
    wy = []
    wz = []
    thrust = []
    for i in range(data_length):
        # if not got_data:
        #     if exp_data[i][12]>0.4:
        #         index = i
        #         print("index", index)
        #         got_data = True
        # if got_data: 
        wx_p.append(exp_data[i][0])
        wy_p.append(exp_data[i][1])
        wz_p.append(exp_data[i][2])
        thrust_p.append(exp_data[i][3])

        wx.append(exp_data[i][4])
        wy.append(exp_data[i][5])
        wz.append(exp_data[i][6])
        thrust.append(exp_data[i][7])

    t = np.arange(0., 0.01*(data_length), 0.01)
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
   
    return wx_p, wy_p, wz_p, thrust_p, wx, wy, wz, thrust, t

def plot_result( with_p, without_p, legend  ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, with_p, 'r', t, without_p, 'b')
    plt.legend(labels=[legend + '_with_p', legend])
    data_range = np.max(with_p) - np.min(with_p)
    if data_range < 0.3:
        maloc = 0.02 
        miloc = 0.01
    elif data_range < 2:
        maloc = 0.2 
        miloc = 0.1
    elif data_range > 2:
        # maloc = float( '%.1f'%(train_y_range/30))
        # miloc = maloc / 2
        maloc = 1 
        miloc = 0.2

    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(maloc))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(miloc))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    title = sys.argv[2]
    plt.title(title + ':' + legend)
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    figures_path = './figures_' + sys.argv[2] + '/'
    if not os.path.exists(figures_path):
        os.makedirs( figures_path )
    fig.savefig( figures_path + sys.argv[2] + '_' + legend + '.png' )

if __name__ == '__main__':
    # np_file = './exp_data_control_with_p_20210908_q330_circle_40s.npy'
    np_file = './'+ sys.argv[3]+ '/compare_control_with_p/'+ sys.argv[1]
    
    wx_p, wy_p, wz_p, thrust_p, wx, wy, wz, thrust, t = load_npy(np_file)

    plot_result(wx_p, wx, 'wx' )
    plot_result(wy_p, wy, 'wy' )
    plot_result(wz_p, wz, 'wz' )
    plot_result(thrust_p, thrust, 'thrust' )
