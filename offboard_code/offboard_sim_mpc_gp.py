#!/usr/bin/env python3
# coding=UTF-8
'''
Author: Yinfeng Long
Date: 2021-08-28
last_edit: 2021-09-19
'''

import numpy as np

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import AccelStamped
from itm_mav_msgs.msg import SetMission

from gpr.mpc_GPyTorch_predict import GpMean, GpMeanCombine 

class GpPredict(object):
    def __init__(self):
        self.rate = rospy.Rate(100)

        file_path = rospy.get_param('/offboard_sim_gp/gp_model')
        npz_name = rospy.get_param('/offboard_sim_gp/npz_name')
        data_type= rospy.get_param('/offboard_sim_gp/data_type')
        self.gp_limit = rospy.get_param('/offboard_sim_gp/gp_limit')
        print("data_type:{}".format(data_type))
        
        if data_type == 'single':
            # load gp model
            self.gpMPCVx = GpMean('vx','y_vx', file_path, npz_name)
            self.gpMPCVy = GpMean('vy','y_vy', file_path, npz_name)
            self.gpMPCVz = GpMean('vz','y_vz', file_path, npz_name)
        elif data_type == 'combine':
            # load gp model
            self.gpMPCVx = GpMeanCombine('vx', file_path, npz_name)
            self.gpMPCVy = GpMeanCombine('vy', file_path, npz_name)
            self.gpMPCVz = GpMeanCombine('vz', file_path, npz_name)

        # subscribers
        robot_odom_sub = rospy.Subscriber('/robot_pose', Odometry, self.robot_odom_callback)
        robot_pose_sub = rospy.Subscriber(
            '/real_pose', PoseStamped, self.robot_pose_callback)
        self.uav_pose = None
        self.got_robot_pose = False
        self.got_robot_odom = False

        self.is_velocity_init = False
        self.current_time = None
        self.current_position = None
        self.previous_time = None
        self.last_position = None
        self.last_velocity = None
        self.vel = None

        # self.att_rate_cmd_sub = rospy.Subscriber(
        #     '/mavros/setpoint_raw/attitude', AttitudeTarget, self.attitude_rate_cmd_callback)
        # self.att_rate_cmd = None

        # self.command_id_sub = rospy.Subscriber( 
        #     '/itm_quadrotor_control/user_command', SetMission, self.command_callback)
        # self.command_id = None 

        # Publisher
        self.gp_mean_acc_pub = rospy.Publisher('/gp_acceleration_world', AccelStamped, queue_size=1)
        self.gp_mean_list = AccelStamped() 
        self.gp_mean_list.header = Header()
        self.gp_mean_init = False

    def robot_odom_callback(self, msg):
        if not self.got_robot_odom:
            self.got_robot_odom = True
        self.uav_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
				msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.linear.y,
				msg.twist.twist.linear.z ])
        
    def robot_pose_callback(self, msg):
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
        else:
            pass

    # def attitude_rate_cmd_callback(self, msg):
    #     # robot_control as [wx, wy, wz, thrust]
    #     self.att_rate_cmd = np.array(
    #         [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])

    # def command_callback(self, msg):
    #     self.command_id = msg.mission_mode 

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

    def gp_mean_publish_process(self,):
        self.gp_mean_list.header.stamp = rospy.Time.now()

        if not self.gp_mean_init:
            self.gp_mean_list.accel.linear.x = 0 
            self.gp_mean_list.accel.linear.y = 0
            self.gp_mean_list.accel.linear.z = 0
            self.gp_mean_init = True
            rospy.loginfo("gp_mean initialize")

        # if self.got_robot_pose or self.got_robot_odom:
        if self.uav_pose is not None:

            # transform velocity to body frame
            v_b = self.world_to_body( \
                np.array([self.uav_pose[7], self.uav_pose[8], self.uav_pose[9]]), self.uav_pose[3:7])
            # gp predict
            gp_vx_b = self.gpMPCVx.predict_mean( np.array([v_b[0]]) )[0]
            gp_vy_b = self.gpMPCVy.predict_mean( np.array([v_b[1]]) )[0]
            gp_vz_b = self.gpMPCVz.predict_mean( np.array([v_b[2]]) )[0]

            # transform velocity to world frame
            gp_v_w = self.body_to_world( \
                np.array([gp_vx_b, gp_vy_b, gp_vz_b]), self.uav_pose[3:7] )
            # if ( abs(np.array(gp_v_w)) < 2.5 ).all():
            if ( abs(np.array(gp_v_w)) < self.gp_limit ).all():
                    self.gp_mean_list.accel.linear.x = gp_v_w[0] 
                    self.gp_mean_list.accel.linear.y = gp_v_w[1]
                    self.gp_mean_list.accel.linear.z = gp_v_w[2]
        else:
            pass
        
        self.gp_mean_acc_pub.publish(self.gp_mean_list)
        self.rate.sleep()

    def world_to_body( self, v_w, q_array):
        v_b = self.v_dot_q(v_w, self.quaternion_inverse(q_array))
        return v_b

    def body_to_world(self, v_b, q_array):
        v_w = self.v_dot_q( v_b, q_array )
        return v_w

    def v_dot_q(self,v, q):
        rot_mat = self.q_to_rot_mat(q)
        return rot_mat.dot(v)

    def q_to_rot_mat(self, q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
        return rot_mat

    def quaternion_inverse(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([w, -x, -y, -z])
        
if __name__ == '__main__':
    rospy.init_node('gp_predict')

    if rospy.has_param('/offboard_sim_gp/mpc_with_gp'):
        with_gp = rospy.get_param(
            '/offboard_sim_gp/mpc_with_gp')
    else:
        with_gp = False

    #######test!!!!!!######
    # with_gp = True
    if with_gp:
        rospy.loginfo("controller with gp predict")
        gpy_predict = GpPredict() 
        while not rospy.is_shutdown():
            gpy_predict.gp_mean_publish_process()
    else:
        rospy.loginfo('controller without gp predict')
    