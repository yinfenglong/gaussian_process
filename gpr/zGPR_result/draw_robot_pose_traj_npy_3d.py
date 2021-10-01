#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-09-02
'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    if get_gp_acc:
        gp_vx_w = []
        gp_vy_w = []
        gp_vz_w = []
    else:
        pass

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
        if get_gp_acc:
            gp_vx_w.append(exp_data[i][13])
            gp_vy_w.append(exp_data[i][14])
            gp_vz_w.append(exp_data[i][15])
        else:
            pass
    t = np.arange(0., 0.01*(data_length), 0.01)
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
   
    if get_gp_acc:
        return x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w
    else:
        return x, y, z, traj_x, traj_y, traj_z, t

def plot_3d():
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig)
    # all trajectories are the same
    ax.plot(traj_x, traj_y, traj_z , c='r', label='trajectory')
    # ax.scatter(traj_x_gp, traj_y_gp, traj_z_gp , c='g', label='GP_trajectory')
    # ax.scatter(traj_x_egp, traj_y_egp, traj_z_egp , c='b', label='EGP_trajectory')
    ax.scatter(x, y, z , c='k', label='q300_20210928_9_without_gp')
    ax.scatter(x_egp, y_egp, z_egp , c='b', label='q300_20210928_10_with_EGP')
    ax.scatter(x_gp, y_gp, z_gp , c='g', label='q300_20210928_11_with_GP')
    plt.legend(labels=['trajectory', 'q300_20210928_9_without_gp', 'q300_20210928_10_with_EGP', 'q300_20210928_11_with_GP'])

    plt.show()

if __name__ == '__main__':
    get_gp_acc = False
    np_file = './q300/without_gp/exp_data_pose_traj_q300_20210928_9_without_gp.npy'
    np_file_EGP = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20210928_10_with_gp_EGP.npy'
    np_file_GP = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20210928_11_with_gp_GP.npy'
    
    x, y, z, traj_x, traj_y, traj_z, t = load_npy(np_file)
    x_egp, y_egp, z_egp, traj_x_egp, traj_y_egp, traj_z_egp, t_egp = load_npy(np_file_EGP)
    x_gp, y_gp, z_gp, traj_x_gp, traj_y_gp, traj_z_gp, t_gp = load_npy(np_file_GP)

    plot_3d()

    # plt.plot(t, x, 'b', t, traj_x, 'r', t_gp, x_gp, 'g-.', t_gp, traj_x_gp, 'k:')
    # plt.legend(labels=['robot_pose', 'robot_traj', 'robot_pose_gp', 'robot_traj_gp' ])
    # plt.title('m=1.709: x')
    # plt.show()
    # plt.plot(t, y, 'b', t, traj_y, 'r', t_gp, y_gp, 'g-.', t_gp, traj_y_gp, 'k:')
    # plt.legend(labels=['robot_pose', 'robot_traj', 'robot_pose_gp', 'robot_traj_gp' ])
    # plt.title('m=1.709: y')
    # plt.show()
    # plt.plot(t, z, 'b', t, traj_z, 'r', t_gp, z_gp, 'g-.', t_gp, traj_z_gp, 'k:')
    # plt.legend(labels=['robot_pose', 'robot_traj', 'robot_pose_gp', 'robot_traj_gp' ])
    # plt.title('m=1.709: z')
    # plt.show()


