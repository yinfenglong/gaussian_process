#!/usr/bin/env python3
# coding=UTF-8
'''
Author: Wei Luo
Date: 2021-05-24 13:12:47
LastEditors: Yinfeng Long
LastEditTime: 2021-08-10 
'''

import rospy
import numpy as np
from itm_mav_msgs.msg import itm_trajectory_msg, itm_trajectory_point
from itm_mav_srvs.srv import itm_trajectory_srv
# from itm_nonlinear_mpc.msg import itm_trajectory_msg, itm_trajectory_point
# from itm_nonlinear_mpc.srv import itm_trajectory_srv
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from random_trajectory_generator import random_matrix_generator


# global parameters
robot_pose = np.array([])
got_robot_pose = False
got_robot_odom = False


def robot_odom_callback(msg):
    global robot_pose, got_robot_odom
    if not got_robot_odom:
        got_robot_odom = True
    robot_pose = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])


def robot_pose_callback(msg):
    global robot_pose, got_robot_pose
    if not got_robot_pose:
        got_robot_pose = True
    robot_pose = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])


if __name__ == '__main__':
    rospy.init_node('uav_squared_trajectory')
    rospy.wait_for_service('/itm_trajectory_srv')
    set_traj_srv = rospy.ServiceProxy('itm_trajectory_srv', itm_trajectory_srv)
    robot_odom_sub_ = rospy.Subscriber(
        '/robot_pose', Odometry, robot_odom_callback)
    robot_pose_sub_ = rospy.Subscriber(
        '/vrpn_client_node/ITM_Q300/pose', PoseStamped, robot_pose_callback)

    # generate random trajectory: x~[-0.75,0.75], y~[-0.75,0.65], z~[0.5,0.65]
    # random_matrix_generator(x_min, x_max, y_min, y_max, z_min, z_max, size)
    trajectory_matrix = random_matrix_generator(-75, 75, -75, 65, 50, 65, 20)
    trajectory_waypoints = np.array( trajectory_matrix )
    # trajectory_waypoints = np.array( 0.01 * np.random.randint( low=[x_min, y_min, z_min], high=[x_max, y_max, z_max], size=(20,3) ) )
    # trajectory_waypoints = 0.01 * np.random.randint( low=[-80,-80,50], high=[80,80,80], size=(50,3) )

    #trajectory_waypoints = np.array([
    #    [0.0, 0.0, 0.5],
    #    [0.5, 0.0, 0.5],
    #    [0.5, 0.5, 0.5],
    #    [0.0, 0.5, 0.5],
    #    [-0.5, 0.5, 0.5],
    #    [-0.5, 0.0, 0.5],
    #    [-0.5, -0.5, 0.5],
    #    [0.0, -0.5, 0.5],
    #    [0.5, -0.5, 0.5],
    #    [0.0, 0.0, 0.5],
    #])

    # main loop
    # trajectory_list = []
    msg_index = 0
    publish_index = 0
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if (not got_robot_pose and not got_robot_odom):
            rospy.logerr("no robot pose or odometry information received")
        elif (got_robot_odom and got_robot_pose):
            rospy.logerr(
                "You've received both odom (sim) and pose (OptiTrack). That is impossible and unsupported! Stop and check your setup")
            break
        else:
            if publish_index == msg_index:
                trajectory_list = []
                current_target = trajectory_waypoints[msg_index]
                point = itm_trajectory_point()
                point.x = current_target[0]
                point.y = current_target[1]
                point.z = current_target[2]
                point.vx = 0.0
                point.vy = 0.0
                point.vz = 0.0
                point.roll = 0.0
                point.pitch = 0.0
                point.yaw = 0.0
                point.derivative = 0
                point.fixed = True
                point.time_known = True
                point.time_stamp = 0.0
                trajectory_list.append(point)
                srv_obj = set_traj_srv(msg_index, trajectory_list)
                if srv_obj.success:
                    publish_index += 1
            # check if arrived the target pose
            pose_error = current_target.flatten() - robot_pose
            if (np.linalg.norm(pose_error) < 0.1):
                msg_index += 1
            else:
                print(np.linalg.norm(pose_error))
                print(current_target)
                print(robot_pose)
        if msg_index == trajectory_waypoints.shape[0]:
            break
        rate.sleep()
