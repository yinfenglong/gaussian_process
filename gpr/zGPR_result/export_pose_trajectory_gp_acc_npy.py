#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yinfeng LOng
Date: 2021-09-02
usage
    roscore
    rosbag play xxx
    python3 export_mission_state_control_npy.py
'''

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, AccelStamped
from mavros_msgs.msg import AttitudeTarget
from itm_mav_msgs.msg import SetMission
# from itm_nonlinear_mpc.msg import itm_trajectory_msg 
from itm_mav_msgs.msg import itm_trajectory_msg
import os.path

class UAVSubNpy(object):
    def __init__(self):
        # pose/odometry
        # self.uav_pose_sub = rospy.Subscriber(
        #     uav_pose_topic, PoseStamped, self.pose_callback)
        self.robot_state_sub = rospy.Subscriber('/robot_pose', Odometry, self.robot_odom_callback)
        self.robot_pose_sub = rospy.Subscriber(
            '/vrpn_client_node/ITM_Q300/pose', PoseStamped, self.robot_pose_callback)

        self.got_robot_pose = False
        self.got_robot_odom = False
        self.uav_pose = None
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
        # self.att_rate_cmd_sub = rospy.Subscriber(
        #     '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_rate_cmd_callback)
        # self.got_cmd = False
        # self.att_rate_cmd = None
        # self.cmd_timer = None

        self.trajectory_sub = rospy.Subscriber(
            '/robot_trajectory', itm_trajectory_msg, self.trajectory_cmd_callback)
        self.got_trajectory_msg = False
        self.uav_trajectory = None
        self.trajectory_timer = None

        self.command_id_sub = rospy.Subscriber( 
            '/itm_quadrotor_control/user_command', SetMission, self.command_callback)
        self.got_id = False
        self.command_id = None 

        # sub parameter p
        # gp_mean_sub = rospy.Subscriber(
        #     '/gp_acceleration_world', AccelStamped, self.gp_mpc_callback)
        gp_mean_sub = rospy.Subscriber(
            '/gp_acc_estimation', AccelStamped, self.gp_mpc_callback)
        self.gp_mean_accel_w = np.array([0, 0, 0]) 

    def robot_odom_callback(self, msg):
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        if not self.got_robot_odom:
            self.got_robot_odom= True
        self.uav_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
				msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.linear.y,
				msg.twist.twist.linear.z ])
        self.pose_timer = rospy.get_time()

    def robot_pose_callback(self, msg):
        self.pose_timer = rospy.get_time()
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        if not self.got_robot_pose:
            self.got_robot_pose = True
        self.current_time = rospy.get_time()
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.velocity_estimation()
        if self.vel is not None:
            if msg.pose.orientation.w > 0:
                self.uav_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, self.vel[0], self.vel[1], self.vel[2] ])
            elif msg.pose.orientation.w < 0:
                self.uav_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        -msg.pose.orientation.w, -msg.pose.orientation.x, -msg.pose.orientation.y,
                        -msg.pose.orientation.z, self.vel[0], self.vel[1], self.vel[2] ])
            rospy.loginfo_once('got_pose')
            rospy.loginfo_once('uav_pose.shape:{}'.format(self.uav_pose.shape))
        else:
            pass

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
                self.vel = 0.2 * self.vel + 0.8 * self.last_velocity

                self.last_velocity = self.vel
                self.previous_time = self.current_time
                self.last_position = self.current_position

    def trajectory_cmd_callback(self, msg):
        self.trajectory_timer = rospy.get_time()
        if not self.got_trajectory_msg:
            self.got_trajectory_msg = True
        temp_traj = msg.traj
        if msg.size != len(temp_traj):
            rospy.logerr('Some data are lost')
        else:
            self.uav_trajectory = np.zeros((1, 3))
            # self.uav_trajectory = np.zeros((msg.size, 3))
            # for i in range(msg.size):
            self.uav_trajectory[0] = np.array([temp_traj[0].x,
                                                temp_traj[0].y,
                                                temp_traj[0].z
                                                # temp_traj[i].vx,
                                                # temp_traj[i].vy,
                                                # temp_traj[i].vz
            ])
            rospy.loginfo_once('got_trajectory')
            rospy.loginfo_once('trajectory.shape:{}'.format(self.uav_trajectory[0].shape))

    def gp_mpc_callback(self, msg):
        # get gp predict value
        self.gp_mean_accel_w = np.array([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z])
        rospy.loginfo_once('got_gp_acc')
        rospy.loginfo_once('gp_acc.shape:{}'.format(self.gp_mean_accel_w.shape))

    # def attitude_rate_cmd_callback(self, msg):
    #     # robot_control as [wx, wy, wz, thrust]
    #     if not self.got_cmd:
    #         self.got_cmd = True
    #     self.att_rate_cmd = np.array(
    #         [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])
    #     self.cmd_timer = rospy.get_time()

    def command_callback(self, msg):
        if not self.got_id:
            self.got_id = True
        self.command_id = msg.mission_mode 
        
if __name__ == '__main__':
    rospy.init_node('uav_analyse')
    rate = rospy.Rate(100)
    is_rate_cmd = True 
    # sub_obj = UAVSubNpy('/vrpn_client_node/ITM_Q330/pose',
    #                     rate_cmd=is_rate_cmd)
    sub_obj = UAVSubNpy( )
    data_list = []

    if is_rate_cmd:
        while not rospy.is_shutdown():
            while sub_obj.uav_trajectory is None or sub_obj.uav_pose is None:
                pass
            current_time_ = rospy.get_time()
            # if current_time_ - sub_obj.trajectory_timer > 3. or current_time_ - sub_obj.pose_timer > 3.:
            if sub_obj.command_id == 2:
                # safe the data
                npy_path = './q300/without_gp/'
                # npy_path = './q300/with_gp/'
                # npy_path = './gazebo/with_gp/'
                if not os.path.exists(npy_path):
                    os.makedirs( npy_path )
                np.save(npy_path + 'exp_data_pose_traj_gp_acc_q300_20211005_11_without_gp.npy', data_list)
                break
            if sub_obj.command_id == 3: 
                # data_list.append(np.append(sub_obj.uav_pose.flatten(),
                #                     sub_obj.uav_trajectory.flatten()))
                # get gp_acc_data
                data_list.append(np.append(np.append(sub_obj.uav_pose.flatten(),
                                    sub_obj.uav_trajectory.flatten()), sub_obj.gp_mean_accel_w.flatten()))
                rospy.loginfo_once('data_list[0].shape):{}'.format(data_list[0].shape))
            rate.sleep()
    else:
        while not rospy.is_shutdown():
            while sub_obj.uav_trajectory is None or sub_obj.uav_pose is None:
                pass
            current_time_ = rospy.get_time()
            if current_time_ - sub_obj.trajectory_timer > 2. or current_time_ - sub_obj.pose_timer > 2.:
                # safe the data
                np.save('exp_data_pose_traj_optitrack.npy', data_list)
                break
            data_list.append(np.append(sub_obj.uav_pose.flatten(),
                                       sub_obj.uav_trajectory.flatten()))
            rate.sleep()

    rospy.loginfo('log accomplish')
