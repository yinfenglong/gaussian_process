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
    # ax.plot(traj_x_gp, traj_y_gp, traj_z_gp , c='r', label='GP_trajectory')
    # ax.scatter(traj_x_egp, traj_y_egp, traj_z_egp , c='r', label='EGP_trajectory')
    # ax.scatter(x, y, z , c='k', label='q300_20210928_9_without_gp')
    ax.scatter(x_egp, y_egp, z_egp , c='b', label='q300_with_EGP')
    ax.scatter(x_agp, y_agp, z_agp , c='g', label='q300_with_AGP')
    # plt.legend(labels=['trajectory', 'q300_20210928_9_without_gp', 'q300_20210928_10_with_EGP', 'q300_20210928_11_with_GP'])
    plt.legend(labels=['trajectory', 'q300_with_EGP', 'q300_with_AGP'])

    plt.show()

def plot_2d(pose, traj, pose_agp, pose_egp, gp_agp, gp_egp, tag ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, pose, 'r-.', t, traj, 'b', t_agp, pose_agp, 'peru', t_agp, gp_agp ,'g:', t_egp, pose_egp, 'm',  t_egp, gp_egp, 'k:')
    plt.legend(labels=['pose', 'traj', 'pose_agp','gp_agp', 'pose_egp', 'gp_egp'])
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
    # title =  #sys.argv[2]
    # plt.title(title + ':' + tag)
    # manger = plt.get_current_fig_manager()
    # manger.window.showMaximized()
    # fig = plt.gcf()
    plt.show()
    # figures_path = './' + folder_name + '/figures_' + np_name + '/'
    # if not os.path.exists(figures_path):
    #     os.makedirs( figures_path )
    # fig.savefig( figures_path + np_name + '_' + tag + '.png' )

def plot_compare(traj, pose_agp1, pose_agp2, gp_agp, gp_egp, tag ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t_agp, traj, 'b', t_agp, pose_agp1, 'peru', t_agp, gp_agp ,'g:', t_egp, pose_agp2, 'm', t_egp, gp_egp, 'k:')
    plt.legend(labels=['traj', '1_pose_gp','1_gp', '2_pose_gp', '2_gp'])
    data_range = np.max(pose_agp1) - np.min(pose_agp1)
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
    # title =  #sys.argv[2]
    # plt.title(title + ':' + tag)
    # manger = plt.get_current_fig_manager()
    # manger.window.showMaximized()
    # fig = plt.gcf()
    plt.show()
    # figures_path = './' + folder_name + '/figures_' + np_name + '/'
    # if not os.path.exists(figures_path):
    #     os.makedirs( figures_path )
    # fig.savefig( figures_path + np_name + '_' + tag + '.png' )

if __name__ == '__main__':
    get_gp_acc = True 
    
    np_file = './q330/without_gp/exp_data_pose_traj_gp_acc_q330_20211008_1_without_gp.npy'
    np_file_AGP = './q330/with_gp/exp_data_pose_traj_gp_acc_q330_20211011_9_with_appgp.npy'
    np_file_EGP = './q330/with_gp/exp_data_pose_traj_gp_acc_q330_20211011_10_with_egp.npy'
    # exp_data_pose_traj_gp_acc_q330_20211011_1_with_appgp
    # exp_data_pose_traj_gp_acc_q330_20211011_2_with_appgp_luo
    # exp_data_pose_traj_gp_acc_q330_20211011_3_with_egp
    # exp_data_pose_traj_gp_acc_q330_20211011_4_with_appgp
    # exp_data_pose_traj_gp_acc_q330_20211011_5_with_appgp
    # exp_data_pose_traj_gp_acc_q330_20211011_6_with_egp
    # exp_data_pose_traj_gp_acc_q330_20211011_7_with_egp
    # exp_data_pose_traj_gp_acc_q330_20211011_8_with_appgp

    # compare
    # np_file_AGP_2 = './q330/with_gp/exp_data_pose_traj_gp_acc_q330_20211008_2_with_appgp.npy'
    # np_file_AGP_4 = './q330/with_gp/exp_data_pose_traj_gp_acc_q330_20211008_4_with_appgp.npy'
    # x_agp, y_agp, z_agp, traj_x_agp, traj_y_agp, traj_z_agp, t_agp = load_npy(np_file_AGP_2)
    # x_egp, y_egp, z_egp, traj_x_egp, traj_y_egp, traj_z_egp, t_egp = load_npy(np_file_AGP_4)
    ###########################

    x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w= load_npy(np_file)
    x_agp, y_agp, z_agp, traj_x_agp, traj_y_agp, traj_z_agp, t_agp, gp_vx_agp, gp_vy_agp, gp_vz_agp = load_npy(np_file_AGP)
    x_egp, y_egp, z_egp, traj_x_egp, traj_y_egp, traj_z_egp, t_egp, gp_vx_egp, gp_vy_egp, gp_vz_egp = load_npy(np_file_EGP)

    # plot_3d()

    # plot_2d(x, traj_x, x_agp, x_egp, gp_vx_agp, gp_vx_egp, 'x' )
    # plot_2d(y, traj_y, y_agp, y_egp, gp_vy_agp, gp_vy_egp, 'y' )
    # plot_2d(z, traj_z, z_agp, z_egp, gp_vz_agp, gp_vz_egp, 'z' )

    # compare
    plot_compare(traj_x_agp, x_agp, x_egp, gp_vx_agp, gp_vx_egp, 'x' )
    plot_compare(traj_y_agp, y_agp, y_egp, gp_vy_agp, gp_vy_egp, 'y' )
    plot_compare(traj_z_agp, z_agp, z_egp, gp_vz_agp, gp_vz_egp, 'z' )
    #############################################

