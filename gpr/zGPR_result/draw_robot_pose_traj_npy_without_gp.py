#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-09-02
last_edit: 2021-09-21

usage: 
    python3 draw_robot_pose_traj_npy_without_gp.py q300/without_gp exp_data_pose_traj_q300_20210920_4_random_20
'''

import numpy as np
from matplotlib import pyplot as plt
import sys
import os.path

def load_npy(np_file):
    exp_data = np.load(np_file, allow_pickle=True)
    data_length = exp_data.shape[0]-2
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
        traj_vx.append(exp_data[i][13])
        traj_vy.append(exp_data[i][14])
        traj_vz.append(exp_data[i][15])
        if get_gp_acc:
            gp_vx_w.append(exp_data[i][16])
            gp_vy_w.append(exp_data[i][17])
            gp_vz_w.append(exp_data[i][18])
        else:
            pass
    t = np.arange(0., 0.01*(data_length), 0.01)
    # t = np.arange(0., 0.01*(data_length-index), 0.01)
   
    if get_gp_acc:
        return x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w
    else:
        return x, y, z, traj_x, traj_y, traj_z, t
        # return x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t

def plot_result( pose, traj, tag ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, pose, 'r', t, traj, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj'])
    data_range = np.max(pose) - np.min(pose)
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
    plt.title(title + ':' + tag)
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    figures_path = './' + sys.argv[1] + '/figures/'
    if not os.path.exists(figures_path):
        os.makedirs( figures_path )
    fig.savefig( figures_path + sys.argv[2] + '_' + tag + '.png' )

if __name__ == '__main__':
    get_gp_acc = True
    # np_file = './exp_data_pose_traj_q330_circle_30s.npy'
    np_file = './' + sys.argv[1] + '/' + sys.argv[2] + '.npy'
    
    if get_gp_acc:
        x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w = load_npy(np_file)
    else:
        x, y, z, traj_x, traj_y, traj_z, t = load_npy(np_file)
    # x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t = load_npy(np_file)
    print("t:{}".format(t.shape))
    print("x:{}".format(len(x)))
    print("traj_x:{}".format(len(traj_x)))

    if get_gp_acc:
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        plt.plot(t, gp_vx_w, 'r-', t, gp_vy_w, 'cyan', t, gp_vz_w, 'b-.')
        plt.legend(labels=['gp_ax_w', 'gp_ay_w', 'gp_az_w'])
        # y_grid
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(axis='y', which='both')
        # x_grid
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(axis='x', which='both')
        plt.title( sys.argv[2] )
        manger = plt.get_current_fig_manager()
        manger.window.showMaximized()
        fig = plt.gcf()
        plt.show()
        fig.savefig( './' + sys.argv[1] + '/figures/' + sys.argv[2] + '.png' )


    # plot_result(x, traj_x, 'x' )
    # plot_result(y, traj_y, 'y' )
    # plot_result(z, traj_z, 'z' )

    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # plt.plot(t, x, 'peru', t, traj_x, 'cyan')
    # plt.plot(t, y, 'm:', t, traj_y, 'k:')
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
    # plt.title( sys.argv[2] + '_xyz.png')
    # manger = plt.get_current_fig_manager()
    # manger.window.showMaximized()
    # fig = plt.gcf()
    # plt.show()
    # fig.savefig( './' + sys.argv[1] + '/figures/' + sys.argv[2] + '_xyz.png' )

