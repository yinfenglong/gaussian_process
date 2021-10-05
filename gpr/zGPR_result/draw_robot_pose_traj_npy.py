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
            if exp_data[i][12]>0.4:
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
    t = np.arange(0., 0.01*(data_length-index), 0.01)
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
    np_file_agp_z = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20211005_5_with_gp_AGP_z_vz.npy'
    # np_file_egp_z = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20211005_7_with_gp_EGP_z_vz.npy'
    np_file_agp = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20211005_6_with_gp_AGP_vz.npy'
    # np_file_egp = './q300/with_gp/exp_data_pose_traj_gp_acc_q300_20211005_8_with_gp_EGP_vz.npy'
    
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

    plt.plot(t, x, 'b', t, traj_x, 'r', t_gp, x_gp, 'g-.', t_gp, traj_x_gp, 'k:')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_agp', 'robot_traj_z_agp', 'robot_pose_agp', 'robot_traj_agp' ])
    plt.title('x')
    plt.show()
    plt.plot(t, y, 'b', t, traj_y, 'r', t_gp, y_gp, 'g-.', t_gp, traj_y_gp, 'k:')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_agp', 'robot_traj_z_agp', 'robot_pose_agp', 'robot_traj_agp' ])
    plt.title('y')
    plt.show()
    plt.plot(t, z, 'b', t, traj_z, 'r', t_gp, z_gp, 'g-.', t_gp, traj_z_gp, 'k:')
    plt.legend(labels=['robot_pose_agp', 'robot_traj_agp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_egp', 'robot_traj_z_egp', 'robot_pose_egp', 'robot_traj_egp' ])
    # plt.legend(labels=['robot_pose_z_agp', 'robot_traj_z_agp', 'robot_pose_agp', 'robot_traj_agp' ])
    plt.title('z')
    plt.show()

    # plt.plot(t, vz, 'r--', t_gp, vz_gp, 'g')
    # plt.legend(labels=['robot_pose', 'robot_pose_gp' ])
    # plt.title('m=1.709: vz')
    # plt.show()
    # plt.plot(t, vz, 'r--', t, traj_vz, 'm--', t_gp, vz_gp, 'g', t_gp, traj_vz_gp, 'b')
    # plt.legend(labels=['robot_pose', 'robot_traj', 'robot_pose_gp', 'robot_traj_gp' ])
    # plt.title('m=1.709: vz')
    # plt.show()
