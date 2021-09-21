#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng LOng
Date: 2021-08-19
usage
    roscore
    python3 offboard_sim_mpc_gp.py
    rosbag play xxx
    python3 scripts/acados/quadrotor_optimizer_q_mpc_q330_set_p.py
    python3 export_mission_control_set_p.py
'''

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelStamped
from mavros_msgs.msg import AttitudeTarget
from itm_mav_msgs.msg import SetMission
import os.path

class UAVSubNpy(object):
    def __init__(self):
        # get u (before setting p)
        self.att_rate_cmd_sub = rospy.Subscriber(
            '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_rate_cmd_callback)
        self.got_cmd = False
        self.att_rate_cmd = None
        self.cmd_timer = None

        # get u (after setting p)
        self.att_setpoint_pub = rospy.Subscriber(
            '/set_p_control', AttitudeTarget, self.set_p_control_callback)
        self.got_control = False
        self.control_with_p = None
        self.control_with_p_timer = None

        self.command_id_sub = rospy.Subscriber( 
            '/itm_quadrotor_control/user_command', SetMission, self.command_callback)
        self.got_id = False
        self.command_id = None 

        # sub parameter p
        gp_mean_sub = rospy.Subscriber(
            '/gp_acceleration_world', AccelStamped, self.gp_mpc_callback)
        self.gp_mean_accel_w = np.array([0, 0, 0]) 
        self.is_gp_init = False
        
    def attitude_rate_cmd_callback(self, msg):
        # robot_control as [wx, wy, wz, thrust]
        if not self.got_cmd:
            self.got_cmd = True
        self.att_rate_cmd = np.array(
            [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])
        self.cmd_timer = rospy.get_time()
    
    def set_p_control_callback(self, msg):
        # robot_control as [wx, wy, wz, thrust]
        if not self.got_control:
            self.got_control = True
        self.control_with_p = np.array(
            [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])
        self.control_with_p_timer = rospy.get_time()

    def command_callback(self, msg):
        if not self.got_id:
            self.got_id = True
        self.command_id = msg.mission_mode 

    def gp_mpc_callback(self, msg):
        # get gp predict value
        if not self.is_gp_init:
            # self.gp_mean_accel_w = np.array([0, 0, 0])
            self.is_gp_init = True
        else:
            self.gp_mean_accel_w = np.array([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z])
    
if __name__ == '__main__':
    rospy.init_node('uav_analyse')
    rate = rospy.Rate(100)
    sub_obj = UAVSubNpy( )
    data_list = []

    while not rospy.is_shutdown():
        while sub_obj.att_rate_cmd is None or sub_obj.control_with_p is None :
            pass
        current_time_ = rospy.get_time()
        if current_time_ - sub_obj.cmd_timer > 2. or current_time_ - sub_obj.control_with_p_timer > 2.:
            # safe the data
            npy_name = rospy.get_param('/offboard_export_control/npy_name')
            folder_name = rospy.get_param('/offboard_export_control/folder_name')
            root_path = '/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/forw_prop/rosbag_npy/'
            npy_path = root_path + folder_name + '/compare_control_with_p/'
            if not os.path.exists(npy_path):
                os.makedirs( npy_path )
            np.save( npy_path + npy_name , data_list)
            
            break
        if sub_obj.command_id == 1 or sub_obj.command_id == 2 or sub_obj.command_id == 5:
            continue 
        elif sub_obj.command_id == 3: 
            rospy.loginfo_once('begin recording')
            data_list.append(np.append(sub_obj.control_with_p.flatten(),
                        sub_obj.att_rate_cmd.flatten()))
            # get gp_acc_data
            # data_list.append(np.append(np.append(sub_obj.control_with_p.flatten(),
            #                        sub_obj.att_rate_cmd.flatten()), sub_obj.gp_mean_accel_w.flatten()))
        rate.sleep()
  

    rospy.loginfo('log accomplish')
