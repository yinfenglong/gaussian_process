#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-09-02
last_edit: 2021-09-23

usage: 
    python3 draw_gp_acc_robot_pose_traj_npy.py q300/with_gp/exp_data_pose_traj_gp_acc_q300_20210923_1_random_0_03.npy
'''

import numpy as np
from matplotlib import pyplot as plt
import sys
import os.path
sys.path.append( os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from gpr_GPyTorch_predict import GpMean
from gpr_GPyTorch_predict_2d import GpMean2d 
from gpr_GPyTorch_dataload_predict import GpMeanApp
from gpr_GPyTorch_approximate_predict_2d import GpMeanApp2d

def load_npy(np_file, model_2d_path, model_path):
    exp_data = np.load(np_file, allow_pickle=True)
    # model_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_low_combine_egp'
    # model_2d_path_low = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_low_combine_egp_2d'
    # model_2d_path_high = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_high_combine_egp_2d'
    
    npz_name = 'data_for_gp_y.npz'
    gpMPCVx = GpMean('vx','y_vx', model_path, npz_name)
    gpMPCVy = GpMean('vy','y_vy', model_path, npz_name)
    # gpMPCVx = GpMean2d('vx','y_vx', 'z', model_2d_path, npz_name)
    # gpMPCVy = GpMean2d('vy','y_vy', 'z', model_2d_path, npz_name)
    gpMPCVz = GpMean2d('vz','y_vz', 'z', model_2d_path, npz_name)

    # Approximate GP Model
    # npz_name = 'data_for_gp_y.npz'
    # gpMPCVx = GpMeanApp('vx','y_vx', model_path, npz_name)
    # gpMPCVy = GpMeanApp('vy','y_vy', model_path, npz_name)
    # gpMPCVx = GpMeanApp2d('vx','y_vx', 'z', model_2d_path, npz_name)
    # gpMPCVy = GpMeanApp2d('vy','y_vy', 'z', model_2d_path, npz_name)
    # gpMPCVz = GpMeanApp2d('vz','y_vz', 'z', model_2d_path, npz_name)

    gp_offline = []
    for data in exp_data:
        v_b = world_to_body(\
            np.array([data[7], data[8], data[9]]), data[3:7])
        # gp predict
        gp_vx_b = gpMPCVx.predict_mean( np.array([v_b[0]]) )[0]
        gp_vy_b = gpMPCVy.predict_mean( np.array([v_b[1]]) )[0]
        # gp_vz_b = gpMPCVz.predict_mean( np.array([v_b[2]]) )[0]
        # x, y, z with vz
        # gp_vx_b = gpMPCVx.predict_mean( np.c_[v_b[0], data[2]] )[0]
        # gp_vy_b = gpMPCVy.predict_mean( np.c_[v_b[1], data[2]] )[0]
        gp_vz_b = gpMPCVz.predict_mean( np.c_[v_b[2], data[2]] )[0]
        # transform velocity to world frame
        gp_v_w = body_to_world( \
            np.array([gp_vx_b, gp_vy_b, gp_vz_b]), data[3:7] )
        gp_offline.append(gp_v_w)

    data_length = exp_data.shape[0]
    print("Data length:", data_length)
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

    if get_gp_acc:
        gp_vx_w = []
        gp_vy_w = []
        gp_vz_w = []
    else:
        pass

    # ref_x = np.array([0.]*t.shape[0])
    for i in range(data_length):
        if not got_data:
            if exp_data[i][10] > 0: #lemniscate
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
            if get_gp_acc:
                gp_vx_w.append(exp_data[i][13])
                gp_vy_w.append(exp_data[i][14])
                gp_vz_w.append(exp_data[i][15])
            else:
                pass
    # t = 0.01 * np.arange(data_length)
    t = 0.01 * np.arange(data_length-index)
    gp_offline = np.array(gp_offline)
    gp_offline = gp_offline[index:,:]
   
    if get_gp_acc:
        return x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w, gp_offline
    else:
        return x, y, z, traj_x, traj_y, traj_z, t, gp_offline

def world_to_body( v_w, q_array):
    v_b = v_dot_q(v_w, quaternion_inverse(q_array))
    return v_b

def body_to_world( v_b, q_array):
    v_w = v_dot_q( v_b, q_array )
    return v_w

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    return rot_mat.dot(v)

def q_to_rot_mat( q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat

def quaternion_inverse( q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([w, -x, -y, -z])

def plot_pose_traj_gp( pose, traj, gp_acc, tag, gp_offline_i ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, pose, 'r', t, traj, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj'])
    ax2 = ax.twinx()
    ax2.plot(t, gp_acc,'m:')
    ax2.plot(t, gp_offline_i, 'k:')
    ax2.legend([
        'gp_offline_low',
        'gp_offline_high',
    ], loc=2)

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
    title = np_name #sys.argv[2]
    plt.title(title + ':' + tag)
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    # figures_path = './' + folder_name + '/figures' + np_name + '/'
    if not os.path.exists(figures_path):
        os.makedirs( figures_path )
    # fig.savefig( figures_path + np_name + '_' + tag + '.png' )
    fig.savefig('../../thesis_figures/svg/' + 'figures' + '_' + tag + '.svg', format='svg', dpi=800 )

def plot_pose_traj_gp_three( pose, traj, gp_acc, tag, gp_offline_i, gp_offline_3 ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, pose, 'r', t, traj, 'b')
    plt.legend(labels=['robot_pose', 'robot_traj'])
    ax2 = ax.twinx()
    ax2.plot(t, gp_acc,'m-.')
    ax2.plot(t, gp_offline_i, 'k:')
    ax2.plot(t, gp_offline_3, 'g--')
    ax2.legend([
        'gp_offline_low',
        'gp_offline_high',
        'gp_offline_combine'
    ], loc=2)
    # ax2.set_ylim( [-0.2,0.2])
    # if tag == 'x':
    #     ax2.set_ylim( [0.05,0.25])
    # elif tag == 'y':
    #     ax2.set_ylim( [-0.25,-0.05])
    # else:
    #     ax2.set_ylim( [-0.4,0.0])
    if tag == 'x':
        ax2.set_ylim( [0.,0.6])
    elif tag == 'y':
        ax2.set_ylim( [-0.5,0.8])
    else:
        ax2.set_ylim( [-0.5,0.3])
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
    title = np_name #sys.argv[2]
    plt.title(title + ':' + tag)
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    # figures_path = './' + folder_name + '/figures' + np_name + '/'
    if not os.path.exists(figures_path):
        os.makedirs( figures_path )
    # fig.savefig( figures_path + np_name + '_' + tag + '.png' )
    fig.savefig('../../thesis_figures/svg/' + 'figures' + '_' + tag + '.svg', format='svg', dpi=800 )

def plot_error():
    # print error_mean, plot error between pose and trajectory
    gp_offline_x = np.array(gp_offline[:,0])
    gp_offline_y = np.array(gp_offline[:,1])
    gp_offline_z = np.array(gp_offline[:,2])
    gp_offline_x_2 = np.array(gp_offline_2[:,0])
    gp_offline_y_2 = np.array(gp_offline_2[:,1])
    gp_offline_z_2 = np.array(gp_offline_2[:,2])
    # error = np.sqrt( (x_arr - traj_x_arr)**2 + (y_arr - traj_y_arr)**2 + (z_arr - traj_z_arr)**2 )
    # error_x = np.abs(gp_offline_x_2 - gp_offline_x)
    error_x = gp_offline_x_2 - gp_offline_x
    error_x_mean = np.mean( error_x )
    print("difference_x_mean:{}".format(error_x_mean))
    # error_y = np.abs(gp_offline_y_2 - gp_offline_y)
    error_y = gp_offline_y_2 - gp_offline_y
    error_y_mean = np.mean( error_y )
    print("difference_y_mean:{}".format(error_y_mean))
    # error_z = np.abs(gp_offline_z_2 - gp_offline_z)
    error_z = gp_offline_z_2 - gp_offline_z
    error_z_mean = np.mean( error_z )
    print("difference_z_mean:{}".format(error_z_mean))
    # error_mean = np.mean(error)
    # print("error_mean:{}".format(error_mean))
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot( t, error_x, 'r', t, error_y, 'b', t, error_z, 'g' )
    # ax.set_ylim( [0,0.3])
    plt.legend(labels=['gp_difference_x', 'gp_difference_y', 'gp_difference_z'])
    plt.title( np_name )
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    # fig.savefig( './' + folder_name + '/error/' + np_name + '.png' 
    fig.savefig('../../thesis_figures/svg/' + 'difference.svg', format='svg', dpi=800 )

if __name__ == '__main__':
    get_gp_acc = False 
    str_argv_1 = sys.argv[1]
    quadrotor_name, if_with_gp, np_file = str_argv_1.split('/')
    np_name, np_suffix = np_file.split('.', 1)
    np_file = './' + quadrotor_name + '/' + if_with_gp + '/' + np_file
    print("quadrotor_name:{}".format(quadrotor_name))
    print("if_with_gp:{}".format(if_with_gp))
    print("np_file: {}".format(np_file))
    print("np_name: {}".format(np_name))
    print("np_suffix: {}".format(np_suffix))
    folder_name = quadrotor_name + '/' + if_with_gp
    figures_path = './' + folder_name + '/figures_' + np_name + '_offline/'
    
    # model_2d_path_low = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_low_combine_appgp_2d'
    # model_2d_path_high = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_high_combine_appgp_2d'
    # model_2d_path_com = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q330/20211021_height_combine_app_2d'
    
    model_2d_path_low = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_low_combine_egp_2d'
    model_2d_path_high = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_high_combine_egp_2d'
    model_2d_path_com = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_mix_combine_egp_2d'

    model_path_low = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_low_combine_egp'
    model_path_high = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_high_combine_egp'
    model_path_com = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q330/20211021_mix_combine_egp'

    # if get_gp_acc:
    #     x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w, gp_offline = load_npy(np_file, model_2d_path_low)
    # else:
    #     x, y, z, traj_x, traj_y, traj_z, t, gp_offline = load_npy(np_file, model_2d_path_low)
    # x_2, y_2, z_2, traj_x_2, traj_y_2, traj_z_2, t_2, gp_offline_2 = load_npy(np_file, model_2d_path_high)
    # x_3, y_3, z_3, traj_x_3, traj_y_3, traj_z_3, t_3, gp_offline_3 = load_npy(np_file, model_2d_path_com)

    x, y, z, traj_x, traj_y, traj_z, t, gp_offline = load_npy(np_file, model_2d_path_low, model_path_low)
    x_2, y_2, z_2, traj_x_2, traj_y_2, traj_z_2, t_2, gp_offline_2 = load_npy(np_file, model_2d_path_high, model_path_high)
    x_3, y_3, z_3, traj_x_3, traj_y_3, traj_z_3, t_3, gp_offline_3 = load_npy(np_file, model_2d_path_com, model_path_com)
    print("t:{}".format(t.shape))
    print("x:{}".format(len(x)))
    print("traj_x:{}".format(len(traj_x)))

    # gp_offline = np.array(gp_offline)
    # gp_offline_2 = np.array(gp_offline_2)
    # gp_offline_3 = np.array(gp_offline_3)
    print("gp_offline.shape:{}".format(gp_offline.shape))

    # compare two offline gp
    # plot_pose_traj_gp(x, traj_x, gp_offline[:,0], 'x', gp_offline_2[:,0])
    # plot_pose_traj_gp(y, traj_y, gp_offline[:,1], 'y', gp_offline_2[:,1])
    # plot_pose_traj_gp(z, traj_z, gp_offline[:,2], 'z', gp_offline_2[:,2])

    # compare three offline gp
    plot_pose_traj_gp_three(x, traj_x, gp_offline[:,0], 'x', gp_offline_2[:,0], gp_offline_3[:,0])
    plot_pose_traj_gp_three(y, traj_y, gp_offline[:,1], 'y', gp_offline_2[:,1], gp_offline_3[:,1])
    plot_pose_traj_gp_three(z, traj_z, gp_offline[:,2], 'z', gp_offline_2[:,2], gp_offline_3[:,2])

    # plot_error()

