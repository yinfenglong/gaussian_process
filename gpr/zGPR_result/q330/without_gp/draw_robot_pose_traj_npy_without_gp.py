#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-09-02
'''

import numpy as np
from matplotlib import pyplot as plt

def load_npy(np_file):
    exp_data = np.load(np_file, allow_pickle=True)
    data_length = exp_data.shape[0]
    print("Data length:", data_length)
    # got_data = False

    # t = np.arange(0., 0.01*data_length, 0.01)
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    traj_x = []
    traj_y = []
    traj_z = []
    traj_vx = []
    traj_vy = []
    traj_vz = []
    # ref_x = np.array([0.]*t.shape[0])
    for i in range(data_length):
        # if not got_data:
        #     if exp_data[i][12]>0.4:
        #         index = i
        #         print("index", index)
        #         got_data = True
        # if got_data: 
        x.append(exp_data[i][0])
        y.append(exp_data[i][1])
        z.append(exp_data[i][2])
        vx.append(exp_data[i][7])
        vy.append(exp_data[i][8])
        vz.append(exp_data[i][9])

        traj_x.append(exp_data[i][10])
        traj_y.append(exp_data[i][11])
        traj_z.append(exp_data[i][12])
        traj_vx.append(exp_data[i][13])
        traj_vy.append(exp_data[i][14])
        traj_vz.append(exp_data[i][15])
    t = np.arange(0., 0.01*(data_length), 0.01)
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
   
    return x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t

if __name__ == '__main__':
    np_file = './exp_data_pose_traj_q330_circle_30s.npy'
    
    x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t = load_npy(np_file)

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, x, 'r', t, traj_x, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj'])
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('q330_circle_30s: x')
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, y, 'r', t, traj_y, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj' ])
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('q330_circle_30s: y')
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, z, 'r', t, traj_z, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj'])
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('q330_circle_30s: z')
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, x, 'peru', t, traj_x, 'cyan')
    plt.plot(t, y, 'g:', t, traj_y, 'k:')
    plt.plot(t, z, 'r-.', t, traj_z, 'b-.')
    plt.legend(labels=['robot_pose_x', 'robot_traj_x', 'robot_pose_y', 'robot_traj_y', 'robot_pose_z', 'robot_traj_z'])
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('q330_circle_30s')
    plt.show()

