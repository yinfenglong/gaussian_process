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

    # t = np.arange(0., 0.01*data_length, 0.01)
    # twist velocity
    vx_t = []
    vy_t = []
    vz_t = []
    # velocity estimation
    vx = []
    vy = []
    vz = []
    # ref_x = np.array([0.]*t.shape[0])
    for i in range(data_length):
        vx_t.append(exp_data[i][0])
        vy_t.append(exp_data[i][1])
        vz_t.append(exp_data[i][2])
        vx.append(exp_data[i][3])
        vy.append(exp_data[i][4])
        vz.append(exp_data[i][5])

    t = np.arange(0., 0.01*(data_length), 0.01)
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
   
    return vx_t, vy_t, vz_t, vx, vy, vz, t 
if __name__ == '__main__':
    np_file = './compare_vel_twist_est.npy'
    # np_file = './compare_vel_twist_est_ohne_lpf.npy'
    
    vx_t, vy_t, vz_t, vx, vy, vz, t = load_npy(np_file)

    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # plt.plot(t, x, 'r', t, traj_x, 'b')
    # plt.legend(labels=['robot_pose', 'robot_traj'])
    # # y_grid
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    # ax.grid(axis='y', which='both')
    # # x_grid
    # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.grid(axis='x', which='both')
    # plt.title('q330_circle_30s: x')
    # plt.show()

    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # plt.plot(t, y, 'r', t, traj_y, 'b')
    # plt.legend(labels=['robot_pose', 'robot_traj' ])
    # # y_grid
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    # ax.grid(axis='y', which='both')
    # # x_grid
    # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.grid(axis='x', which='both')
    # plt.title('q330_circle_30s: y')
    # plt.show()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, vz, 'r', t, vz_t, 'b')
    plt.legend(labels=['estimated_velocity', 'twist_velocity'])
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('compare velocity of twist and estimation: vz')
    plt.show()

    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # plt.plot(t, x, 'peru', t, traj_x, 'cyan')
    # plt.plot(t, y, 'g:', t, traj_y, 'k:')
    # plt.plot(t, z, 'r-.', t, traj_z, 'b-.')
    # plt.legend(labels=['robot_pose_x', 'robot_traj_x', 'robot_pose_y', 'robot_traj_y', 'robot_pose_z', 'robot_traj_z'])
    # # y_grid
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    # ax.grid(axis='y', which='both')
    # # x_grid
    # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.grid(axis='x', which='both')
    # plt.title('q330_circle_30s')
    # plt.show()

