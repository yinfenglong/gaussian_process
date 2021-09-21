#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng LOng
Date: 2021-08-19
usage
    roscore
    rosbag play xxx
    python3 export_mission_state_control_npy.py
'''

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import AttitudeTarget
from itm_mav_msgs.msg import SetMission

class UAVSubNpy(object):
    def __init__(self):
        # pose/odometry
        # self.uav_pose_sub = rospy.Subscriber(
        #     uav_pose_topic, PoseStamped, self.pose_callback)
        self.vel_twist_sub = rospy.Subscriber('/robot_pose', Odometry, self.vel_twist_callback)
        self.vel_twist = None
        self.got_twist_msg = False
        self.twist_timer = None

        self.vel_est_sub = rospy.Subscriber('/robot_pose', Odometry, self.vel_est_callback)
        self.vel_est = None
        self.got_vel_est = False
        self.est_timer = None
        
        self.is_velocity_init = False
        self.current_time = None
        self.current_position = None
        self.previous_time = None
        self.last_position = None
        self.last_velocity = None
        self.vel = None

        self.command_id_sub = rospy.Subscriber( 
            '/itm_quadrotor_control/user_command', SetMission, self.command_callback)
        self.got_id = False
        self.command_id = None 

    def vel_twist_callback(self, msg):
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        if not self.got_twist_msg:
            self.got_twist = True
        self.vel_twist = np.array( [msg.twist.twist.linear.x, msg.twist.twist.linear.y,
				msg.twist.twist.linear.z ])
        self.twist_timer = rospy.get_time()

    def vel_est_callback(self, msg):
        if not self.got_vel_est:
            self.got_vel_est = True
        self.current_time = rospy.get_time()
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.velocity_estimation()
        if self.vel is not None:
            self.vel_est = np.array([self.vel[0], self.vel[1], self.vel[2]])
        else:
            pass
        self.est_timer = rospy.get_time()   

    def velocity_estimation(self, ):
        if not self.is_velocity_init:
            self.is_velocity_init = True
            self.last_position = self.current_position
            self.previous_time = self.current_time
            self.last_velocity = np.array([0., 0., 0.])
        else:
            dt = self.current_time - self.previous_time
            if dt>=0.01:
                self.vel = (self.current_position - self.last_position)/(1e-5 + dt)
                # self.vel = 0.8 * self.vel + 0.2 * self.last_velocity
                # self.vel = 0.6 * self.vel + 0.4 * self.last_velocity
                self.vel = 0.2 * self.vel + 0.8 * self.last_velocity

                self.last_velocity = self.vel
                self.previous_time = self.current_time
                self.last_position = self.current_position

    def command_callback(self, msg):
        if not self.got_id:
            self.got_id = True
        self.command_id = msg.mission_mode 
        
if __name__ == '__main__':
    rospy.init_node('uav_analyse')
    rate = rospy.Rate(100)
    is_rate_cmd = True 
    # sub_obj = UAVSubNpy('/vrpn_client_node/ITM_Q300/pose',
    #                     rate_cmd=is_rate_cmd)
    sub_obj = UAVSubNpy( )
    data_list = []

    if is_rate_cmd:
        while not rospy.is_shutdown():
            while sub_obj.vel_twist is None or sub_obj.vel_est is None:
                pass
            current_time_ = rospy.get_time()
            if current_time_ - sub_obj.twist_timer > 2. or current_time_ - sub_obj.est_timer > 2.:
                # safe the data
                # np.save('./q330/exp_data_20210913_1_hover.npy', data_list)
                np.save('./compare_vel_twist_est.npy', data_list)
                break
            if sub_obj.command_id == 1 or sub_obj.command_id == 2 or sub_obj.command_id == 5:
                continue 
            elif sub_obj.command_id == 3: 
            # take_off, hover, land
            # if sub_obj.command_id == 1 or sub_obj.command_id == 2 or sub_obj.command_id == 4:
            #     continue 
            # elif sub_obj.command_id == 5: 
                data_list.append(np.append(sub_obj.vel_twist.flatten(),
                                       sub_obj.vel_est.flatten()))
            rate.sleep()
    else:
        while not rospy.is_shutdown():
            while sub_obj.att_cmd is None or sub_obj.uav_pose is None:
                pass
            current_time_ = rospy.get_time()
            if current_time_ - sub_obj.cmd_timer > 2. or current_time_ - sub_obj.pose_timer > 2.:
                # safe the data
                np.save('./q330/exp_data_v_est.npy', data_list)
                break
            data_list.append(np.append(sub_obj.uav_pose.flatten(),
                                       sub_obj.att_cmd.flatten()))
            rate.sleep()

    rospy.loginfo('log accomplish')
