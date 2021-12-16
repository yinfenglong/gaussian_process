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
    got_data = False

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
        if not got_data:
            if exp_data[i][11]>0:
                index = i
                print("index", index)
                got_data = True
        if got_data: 
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
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
    t = 0.01 * np.arange(data_length-index)
    # traj_x = []
    # traj_y = []
    # traj_z = []
    # traj_vx = []
    # traj_vy = []
    # traj_vz = []
    # for i in range(data_length):
    #     traj_x.append(exp_data[i][10])
    #     traj_y.append(exp_data[i][11])
    #     traj_z.append(exp_data[i][12])
    #     traj_vx.append(exp_data[i][13])
    #     traj_vy.append(exp_data[i][14])
    #     traj_vz.append(exp_data[i][15])
    return x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t

if __name__ == '__main__':
    #1.709kg, random traj
    # with gp
    # np_file_agp_z = './gazebo/without_gp/exp_data_pose_traj_gp_offset_new_u.npy'
    # without gp
    # np_file_agp = './gazebo/without_gp/exp_data_pose_traj_offset_new_u.npy'
    #1.709kg, circle
    np_file_agp_z = './gazebo/with_gp/exp_data_pose_traj_gp_acc_gazebo_20211117_2_with_gp.npy'
    np_file_agp = './gazebo/without_gp/exp_data_pose_traj_gp_acc_gazebo_20211117_1_without_gp.npy'
    
    x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t = load_npy(np_file_agp_z)
    x_gp, y_gp, z_gp, vx_gp, vy_gp, vz_gp, traj_x_gp, traj_y_gp, traj_z_gp, traj_vx_gp, traj_vy_gp, traj_vz_gp, t_gp = load_npy(np_file_agp)

    # plt.plot(t, x, 'r--', t, traj_x, 'b--')
    # plt.legend(labels=['robot_pose:x', 'robot_traj:x' ])
    # plt.title('m=1.709, mpc without gp')
    # plt.show()
    # plt.plot(t, y, 'r--', t, traj_y, 'b--')
    # plt.legend(labels=['robot_pose:y', 'robot_traj:y' ])
    # plt.title('m=1.709, mpc without gp')
    # plt.show()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, x, 'g-.', t, traj_x, 'b', t_gp, x_gp, 'r--', t_gp, traj_x_gp, 'k:')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_agp', 'robot_traj_z_agp', 'robot_pose_agp', 'robot_traj_agp' ])
    maloc = 0.1 
    miloc = 0.05
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(maloc))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(miloc))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    plt.title('x')
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    fig.savefig('../../thesis_figures/svg/' + 'figures_x.svg', format='svg', dpi=800 )

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, y, 'g-.', t, traj_y, 'b', t_gp, y_gp, 'r--', t_gp, traj_y_gp, 'k:')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    maloc = 0.1 
    miloc = 0.05
    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(maloc))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(miloc))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    # t.title('y')
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    fig.savefig('../../thesis_figures/svg/' + 'figures_y.svg', format='svg', dpi=800 )

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, z, 'g-.', t, traj_z, 'b', t_gp, z_gp, 'r--', t_gp, traj_z_gp, 'k-.')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_agp', 'robot_traj_z_agp', 'robot_pose_agp', 'robot_traj_agp' ])
    data_range = np.max(z_gp) - np.min(z_gp)
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
    plt.title('z')
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    fig.savefig('../../thesis_figures/svg/' + 'figures_z.svg', format='svg', dpi=800 )

    # plt.plot(t, vz, 'r--', t_gp, vz_gp, 'g')
    # plt.legend(labels=['robot_pose', 'robot_pose_gp' ])
    # plt.title('m=1.709: vz')
    # plt.show()
    # plt.plot(t, vz, 'r--', t, traj_vz, 'm--', t_gp, vz_gp, 'g', t_gp, traj_vz_gp, 'b')
    # plt.legend(labels=['robot_pose', 'robot_traj', 'robot_pose_gp', 'robot_traj_gp' ])
    # plt.title('m=1.709: vz')
    # plt.show()
