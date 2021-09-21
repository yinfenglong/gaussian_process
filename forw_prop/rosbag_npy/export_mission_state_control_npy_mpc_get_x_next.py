#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng LOng
Date: 2021-08-19
usage
    roscore
    rosbag play xxx
    python3 scripts/acados/quadrotor_optimizer_q_mpc_get_x_next.py
    python3 export_mission_state_control_npy_mpc_get_x_next.py
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
        self.robot_state_sub = rospy.Subscriber('/robot_pose', Odometry, self.robot_odom_callback)
        self.robot_pose_sub = rospy.Subscriber(
            '/vrpn_client_node/ITM_Q330/pose', PoseStamped, self.robot_pose_callback)
        self.uav_pose = None
        self.got_robot_pose = False
        self.got_robot_odom = False
        self.pose_timer = None
        
        self.is_velocity_init = False
        self.current_time = None
        self.current_position = None
        self.previous_time = None
        self.last_position = None
        self.last_velocity = None
        self.vel = None

        # if rate_cmd:
        #     # attitude rate
        #     self.att_rate_cmd_sub = rospy.Subscriber(
        #         '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_rate_cmd_callback)
        # else:
        #     self.att_rate_cmd_sub = rospy.Subscriber(
        #         '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_cmd_callback)
        self.att_rate_cmd_sub = rospy.Subscriber(
            '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_rate_cmd_callback)
        self.got_cmd = False
        self.att_rate_cmd = None
        self.cmd_timer = None

        self.command_id_sub = rospy.Subscriber( 
            '/itm_quadrotor_control/user_command', SetMission, self.command_callback)
        self.got_id = False
        self.command_id = None 
        
        # pub mpc x_next
        self.mpc_x_next_sub = rospy.Subscriber(
            '/mpc_x_next_state', Odometry, self.mpc_state_callback)
        self.mpc_next_state = None

    def robot_odom_callback(self, msg):
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        if not self.got_robot_odom:
            self.got_robot_odom = True
        self.uav_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
				msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.linear.y,
				msg.twist.twist.linear.z ])
        self.pose_timer = rospy.get_time()
    
    def robot_pose_callback(self, msg):
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        if not self.got_robot_pose:
            self.got_robot_pose = True
        self.current_time = rospy.get_time()
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.velocity_estimation()
        if self.vel is not None:
            self.uav_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
                    msg.pose.orientation.z, self.vel[0], self.vel[1], self.vel[2] ])
        else:
            pass
        self.pose_timer = rospy.get_time()

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
                self.vel = 0.8 * self.vel + 0.2 * self.last_velocity

                self.last_velocity = self.vel
                self.previous_time = self.current_time
                self.last_position = self.current_position


    def attitude_rate_cmd_callback(self, msg):
        # robot_control as [wx, wy, wz, thrust]
        if not self.got_cmd:
            self.got_cmd = True
        # verify u[3]
        self.att_rate_cmd = np.array(
            # [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, (msg.thrust + 0.59)*9.8066])
            [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])
        self.cmd_timer = rospy.get_time()

    def command_callback(self, msg):
        if not self.got_id:
            self.got_id = True
        self.command_id = msg.mission_mode 
    
    def mpc_state_callback(self, msg):
        self.mpc_next_state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
				msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.linear.y,
				msg.twist.twist.linear.z ])
    
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
            while sub_obj.att_rate_cmd is None or sub_obj.uav_pose is None or sub_obj.mpc_next_state is None:
                pass
            current_time_ = rospy.get_time()
            if current_time_ - sub_obj.cmd_timer > 2. or current_time_ - sub_obj.pose_timer > 2.:
                # safe the data
                np.save('./q330/mpc_compare_forw/exp_data_pose_control_mpc_circle_40.npy', data_list)
                break
            if sub_obj.command_id == 1 or sub_obj.command_id == 2 or sub_obj.command_id == 5:
                continue 
            elif sub_obj.command_id == 3: 
                data_list.append(np.append(np.append(sub_obj.uav_pose.flatten(),
                                       sub_obj.att_rate_cmd.flatten()), sub_obj.mpc_next_state.flatten()))
                                    #    sub_obj.mpc_next_state.flatten()))
                # data_list.append(sub_obj.mpc_next_state.flatten())
            rate.sleep()
    else:
        while not rospy.is_shutdown():
            while sub_obj.att_cmd is None or sub_obj.uav_pose is None:
                pass
            current_time_ = rospy.get_time()
            if current_time_ - sub_obj.cmd_timer > 2. or current_time_ - sub_obj.pose_timer > 2.:
                # safe the data
                np.save('exp_data_mpc_control_offset.npy', data_list)
                break
            data_list.append(np.append(sub_obj.uav_pose.flatten(),
                                       sub_obj.att_cmd.flatten(),
                                       sub_obj.mpc_next_state.flatten()))
            rate.sleep()

    rospy.loginfo('log accomplish')
