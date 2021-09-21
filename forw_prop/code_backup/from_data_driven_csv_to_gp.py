#!/usr/bin/env python3
# coding=utf-8
'''
Date: 12.08.2021
Author: Yinfeng Long 
usage 
    python3 forw_prop.py filename.csv 'RK4'/'mpc' points
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

def datas_get( csv_name ):
    data = pd.read_csv( csv_name ) 
    dt = np.array( data.iloc[:, 7] )
    x_out = np.array( data.iloc[:, 4] )
    x_pred = np.array( data.iloc[:, 5] )
    # print( type(x_k_1_mpc) )
    print("dt", dt[1])
    print("x_out:", x_out[1])
    print("x_pred", x_pred[1])
    return dt, x_out, x_pred 

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

# ### convert velocity from word frame to body frame ### #
def world_to_body_velocity_mapping(state_sequence):
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
    # traj = np.array(traj)
    traj = data_numpy( traj )
    print("traj shape", traj.shape)
    p_traj = traj[:,:3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    return [p_traj, a_traj, v_traj]

def data_numpy(array):
    x = []
    for elem in array:
        a = elem.split('[')[1].split(']')[0].split(',')
        a = [float(num) for num in a]
        x = x + [a]
    return np.array(x)

# compare mpc and forward propagation
def compare_mpc_forw(x_mpc, x_pred):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    error = x_mpc - x_pred
    ax.plot(x_mpc[:,0], error[:,0], 'r*')
    ax.plot(x_mpc[:,1], error[:,1], 'g*')
    ax.plot(x_mpc[:,2], error[:,2], 'b*')
    # ax.plot(x_mpc[:,7], error[:,7], 'k*')
    # ax.plot(x_mpc[:,8], error[:,8], 'y*')
    # ax.plot(x_mpc[:,9], error[:,9], 'm*')
    # ax.legend(['x', 'y', 'z', 'vx', 'vy', 'vz'])
    # ax.set_ylim( [-2, 2])
    ax.legend(['x', 'y', 'z'])
    ax.set_ylim( [-0.2, 0.2])
    plt.show()

if __name__ == '__main__':
    # ### x_k is from gazebo or optitrack ### #
    dt, x_out, x_pred = datas_get( sys.argv[1] )
    size = dt.shape[0]
    # drone_model = QuadRotorModel()
    # drone_f = drone_model.f

    # ###forward prop### #
    # for i in range( size ):
    #     x_k_1_pred = RK_4( x_k[i], u_k[i], dt[i])
    #     if i==0:
    #         x_next_pred = x_k_1_pred
    #     else:
    #         x_next_pred = np.concatenate( (x_next_pred, x_k_1_pred), axis=0)

    x_vb_out = world_to_body_velocity_mapping(x_out)
    x_vb_pred = world_to_body_velocity_mapping(x_pred)
 
    '''
    # training data
             gazebo/OptiTrack    forward_prop
    i-1:                          x_vb_pred      
    i:   dt  x_vb_out(=x_train) 
    '''
    x_train = []
    y_train = []
    for i in range(size):
        if (dt[i]!=0):
            y_train.append( (x_vb_out[i] - x_vb_pred[i])/dt[i] )
            x_train.append( x_vb_out[i] )

    # print(x_next_pred.shape)
    # print(x_next_pred[1000:1010])
    # ### save datasets for GP Train ### #
    x_train = np.array( x_train )
    y_train = np.array( y_train )
    print("x_train and y_train shape", x_train.shape, y_train.shape)
    # save: x_train,y_train: x, y, z, Vx, Vy, Vz
    print("store data_driven datas for gp")
    np.savez('/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/GPR/data_driven_gazebo/data_driven_gazebo_for_gp.npz', \
            x=x_train[:,0], y=x_train[:,1], z=x_train[:,2], vx=x_train[:,7], vy=x_train[:,8], vz=x_train[:,9], \
                qw=x_train[:,3], qx=x_train[:,4], qy=x_train[:,5], qz=x_train[:,6],\
            y_x=y_train[:,0], y_y=y_train[:,1], y_z=y_train[:,2], \
            y_qw = y_train[:,3], y_qx=y_train[:,4], y_qy=y_train[:,5], y_qz=y_train[:,6],\
            y_vx=y_train[:,7], y_vy=y_train[:,8], y_vz=y_train[:,9])

    # compare_mpc_forw(x_k_1_mpc,x_next_pred)

