#!/usr/bin/env python3
# coding=utf-8
'''
Date: 19.08.2021
Author: Yinfeng Long 
usage 
    python3 forw_prop_npy_update_u.py xxx(.npy)(from rosbag)) 
    python3 forw_prop_npy_update_u_update_offset.py q330/20210913_4_20_random 
'''

import sys
import pandas as pd
import numpy as np
import sys
import os.path
sys.path.append( os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from acados.quadrotor_model_q import QuadRotorModel
from matplotlib import pyplot as plt
import casadi as ca

def npy_get():
    # from npy load datas
    # np_file_root = '/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/forw_prop/rosbag_npy/'
    # np_file_root = '/home/itm_stud/test_ma_ws/MA_Experiment/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/forw_prop/rosbag_npy/'
    np_file_root = './rosbag_npy/'
    np_file = np_file_root + sys.argv[1] + '.npy' 
    # control_offset = float( sys.argv[2] )
    # x, y, z, qw, qx, qy, qz, vx, vy, vz; wx, wy, wz, thrust; thrust_compensate(because of battery)
    exp_data = np.load(np_file, allow_pickle=True)
    data_length = exp_data.shape[0]
    x_k = []
    u_k = []
    com_thrust_k = []

    for i in range(data_length):
        x_k.append(exp_data[i][0:10])
        u_k.append(exp_data[i][10:14])
        com_thrust_k.append(exp_data[i][14])
    u_k = np.array(u_k)
    print("u_k[0] before update u:", u_k[0])
    print("initial thrust:", com_thrust_k[0])
    # update u_k     
    for i in range(data_length):
        u_k[i,3] = (u_k[i,3]/ com_thrust_k[i])*9.8066
    x_k = np.array(x_k)
    print("u_k", u_k.shape)
    print("x_k", x_k.shape)
    print("u_k[0]", u_k[0])
    print("x_k[0]", x_k[0])
    return u_k, x_k, data_length 

def RK_4( s_t_, c_, dt_):
    # discretize Runge Kutta 4
    ## approach 1
    k1 = drone_f(s_t_, c_ )
    k2 = drone_f(s_t_+ dt_/2.0*k1, c_ )
    k3 = drone_f(s_t_+ dt_/2.0*k2, c_ )
    k4 = drone_f(s_t_+dt_*k3, c_ )
    ## approach 2
    # k1 = self.dyn_function(s_t_, c_, f_)
    # k2 = self.dyn_function(s_t_+self.Ts/2.0*k1, c_, f_)
    # k3 = self.dyn_function(s_t_+self.Ts/2.0*k2, c_, f_)
    # k4 = self.dyn_function(s_t_+self.Ts*k3, c_, f_)

    result_ = s_t_ + dt_/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return np.transpose(result_)

def world_to_body_velocity_mapping(state_sequence):
    # ### convert velocity from word frame to body frame ### #
    """
    :param state_sequence: N x 13 state array, where N is the number of states in the sequence.
    :return: An N x 13 sequence of states, but where the velocities (assumed to be in positions 7, 8, 9) have been
    rotated from world to body frame. The rotation is made using the quaternion in positions 3, 4, 5, 6.
    """

    p, q, v_w = separate_variables(state_sequence)
    v_b = []
    for i in range(len(q)):
        v_b.append( v_dot_q( v_w[i], quaternion_inverse(q[i]) ) )
    v_b = np.stack(v_b)
    return np.concatenate((p, q, v_b), 1)
    # return v_b

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return ca.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = ca.vertcat(
            ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return ca.vertcat(w, -x, -y, -z)

def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.
    :param traj: N x 10 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """
    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    return [p_traj, a_traj, v_traj]

def show_gazebo_rk4():
    # show results of gazebo and rk4
    f,ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot( x_vb_out[:, 2], 'b*')
    ax.plot( x_vb_pred[:,2], 'r*')
    plt.show()

if __name__ == '__main__':
    ### x_k is from gazebo or optitrack ### #
    u_k, x_k, size = npy_get()
    drone_model = QuadRotorModel()
    drone_f = drone_model.f
    
    # ###forward prop### #
    for i in range( size ):
        x_k_1_pred = RK_4( x_k[i], u_k[i], 0.01) # dt of RK4 is 0.01s
        # if x_k_1_pred[0,2]>1.5:
        #     print("x_k, u_k", x_k[i], u_k[i], x_k_1_pred)
        if i==0:
            x_next_pred = x_k_1_pred
        else:
            x_next_pred = np.concatenate( (x_next_pred, x_k_1_pred), axis=0)

    # transform velocity to body frame
    x_vb_out = world_to_body_velocity_mapping(x_k)
    x_vb_pred = world_to_body_velocity_mapping(x_next_pred)

    # training data
    #               gazebo/OptiTrack    forward_prop
    # i-1:                              x_vb_pred      
    # i:   dt=0.01  x_vb_out(=x_train) 

    y_train = []
    for i in range(1,size):
        y_train.append( x_vb_out[i] - x_vb_pred[i-1] )
        y_train[i-1] /= 0.01

    # print(x_next_pred.shape)
    # print(x_next_pred[1000:1010])
    # ### save datasets for GP Train ### #
    x_train = np.array( x_vb_out[1:, :] )
    y_train = np.array( y_train )
    print(x_train.shape, y_train.shape)

    # save: x_train,y_train: x, y, z, Vx, Vy, Vz
    # datas of mpc or rk4
    # gp_path = '/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/gpr/' + sys.argv[1] + '/' 
    # gp_path = '/home/itm_stud/test_ma_ws/MA_Experiment/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/gpr/' + sys.argv[1] + '/' 
    gp_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + '/gpr/' + sys.argv[1] + '/'
    # print('last_root', os.path.join(os.path.join(os.path.dirname(__file__), '..')) )
    if not os.path.exists(gp_path):
        os.makedirs( gp_path + 'train_pre_model')
        os.makedirs( gp_path + 'figures')
    np.savez(gp_path + 'data_for_gp_y.npz', \
            x=x_train[:,0], y=x_train[:,1], z=x_train[:,2], vx=x_train[:,7], vy=x_train[:,8], vz=x_train[:,9], \
        y_x=y_train[:,0], y_y=y_train[:,1], y_z=y_train[:,2], \
            y_vx=y_train[:,7], y_vy=y_train[:,8], y_vz=y_train[:,9])

    print("store datas of forward propagation")
    # show_gazebo_rk4()


