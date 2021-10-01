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
from gpr_GPyTorch_dataload_predict import GpMeanCombine

def load_npy(np_file):
    exp_data = np.load(np_file, allow_pickle=True)
    # model_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q300/20210928_combine_4_random_ExactGPModel'
    # model_2d_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    # '/q300/20210928_combine_4_random_ExactGPModel_2d'
    # npz_name = 'data_for_gp_y.npz'
    # gpMPCVx = GpMean('vx','y_vx', model_path, npz_name)
    # gpMPCVy = GpMean('vy','y_vy', model_path, npz_name)
    # gpMPCVz = GpMean2d('vz','y_vz','z', model_2d_path, npz_name)

    # Approximate GP Model
    model_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
    '/q300/20210928_combine_4_random_GPModel'
    npz_name = 'data_for_gp_y.npz'
    gpMPCVx = GpMeanCombine('vx','y_vx', model_path, npz_name)
    gpMPCVy = GpMeanCombine('vy','y_vy', model_path, npz_name)
    gpMPCVz = GpMeanCombine('vz','y_vz', model_path, npz_name)

    gp_offline = []
    for data in exp_data:
        v_b = world_to_body(\
            np.array([data[7], data[8], data[9]]), data[3:7])
        # gp predict
        gp_vx_b = gpMPCVx.predict_mean( np.array([v_b[0]]) )[0]
        gp_vy_b = gpMPCVy.predict_mean( np.array([v_b[1]]) )[0]
        gp_vz_b = gpMPCVz.predict_mean( np.array([v_b[2]]) )[0]
        # exact gp model: z with vz
        # gp_vz_b = gpMPCVz.predict_mean( np.c_[v_b[2], data[2]] )[0]
        # transform velocity to world frame
        gp_v_w = body_to_world( \
            np.array([gp_vx_b, gp_vy_b, gp_vz_b]), data[3:7] )
        gp_offline.append(gp_v_w)

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
        return x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w, gp_offline
    else:
        return x, y, z, traj_x, traj_y, traj_z, t

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
    plt.plot(t, pose, 'r', t, traj, 'b', t, gp_acc, 'k', t, gp_offline_i, 'g')
    plt.legend(labels=['robot_pose', 'robot_traj', 'gp_acc', 'EGP_include_z'])
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
    fig.savefig( figures_path + np_name + '_' + tag + '.png' )

if __name__ == '__main__':
    get_gp_acc = True 
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
    
    if get_gp_acc:
        x, y, z, traj_x, traj_y, traj_z, t, gp_vx_w, gp_vy_w, gp_vz_w, gp_offline = load_npy(np_file)
    else:
        x, y, z, traj_x, traj_y, traj_z, t = load_npy(np_file)
    # x, y, z, vx, vy, vz, traj_x, traj_y, traj_z, traj_vx, traj_vy, traj_vz, t = load_npy(np_file)
    print("t:{}".format(t.shape))
    print("x:{}".format(len(x)))
    print("traj_x:{}".format(len(traj_x)))

    gp_offline = np.array(gp_offline)
    print("gp_offline.shape:{}".format(gp_offline.shape))
    plot_pose_traj_gp(x, traj_x, gp_vx_w, 'x', gp_offline[:,0])
    plot_pose_traj_gp(y, traj_y, gp_vy_w, 'y', gp_offline[:,1])
    plot_pose_traj_gp(z, traj_z, gp_vz_w, 'z', gp_offline[:,2])

