#!/usr/bin/env python3
# coding=UTF-8
'''
Author: Wei Luo
Date: 2021-03-21 22:27:50
LastEditors: Yinfeng Long
LastEditTime: 2021-08-30 
'''
import os
import sys
from quadrotor_model_q_set_p import QuadRotorModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca
import scipy.linalg
import numpy as np
import time

import rospy
from geometry_msgs.msg import Twist, PoseStamped, AccelStamped
# from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from threading import Thread
from std_msgs.msg import Header
# from itm_nonlinear_mpc.msg import itm_trajectory_msg
from itm_mav_msgs.msg import itm_trajectory_msg
from mavros_msgs.msg import AttitudeTarget

class MPC_controller(object):
    def __init__(self, quad_model, quad_constraints, t_horizon, n_nodes, sim_required=False):
        self.model = quad_model
        self.constraints = quad_constraints
        self.g_ = 9.8066
        self.T = t_horizon
        self.N = n_nodes
        self.simulation_required = sim_required

        self.current_pose = None
        self.current_state = None
        self.dt = 0.05
        self.rate = rospy.Rate(1 / self.dt)
        self.time_stamp = None
        self.trajectory_path = None
        self.current_twist = np.zeros(3)
        self.att_command = AttitudeTarget()
        self.att_command.type_mask = 128

        self.mpc_x_next_state = Odometry()
        # subscribers
        # the robot state
        robot_state_sub_ = rospy.Subscriber(
            '/robot_pose', Odometry, self.robot_state_callback)
        robot_pose_sub = rospy.Subscriber(
            '/real_pose', PoseStamped, self.robot_pose_callback)
        self.is_velocity_init = False
        self.current_time = None
        self.current_position = None
        self.previous_time = None
        self.last_position = None
        self.last_velocity = None
        self.vel = None

        # sub parameter p
        gp_mean_sub = rospy.Subscriber(
            '/gp_acceleration_world', AccelStamped, self.gp_mpc_callback)
        self.gp_mean_accel_w = None 
        self.is_gp_init = False

        # trajectory
        robot_trajectory_sub_ = rospy.Subscriber('/robot_trajectory', itm_trajectory_msg, self.trajectory_command_callback)
        # publisher
        # pub mpc x_next
        self.mpc_x_next_pub = rospy.Publisher(
            '/mpc_x_next_state', Odometry, queue_size=10)
        # get u (after setting p)
        self.att_setpoint_pub = rospy.Publisher(
            '/set_p_control', AttitudeTarget, queue_size=1)
        self.control_offset = rospy.get_param('/offboard_mpc_controller/control_offset')
        # self.att_setpoint_pub = rospy.Publisher(
        #     '/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        # create a server
        server_ = rospy.Service('uav_mpc_server', SetBool, self.state_server)
        # setup optimizer
        self.quadrotor_optimizer_setup()

        # # It seems that thread cannot ensure the performance of the time
        self.att_thread = Thread(target=self.send_command, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

    def gp_mpc_callback(self, msg):
        # get gp predict value
        if not self.is_gp_init:
            self.gp_mean_accel_w = np.array([0, 0, 0])
            self.is_gp_init = True
        else:
            self.gp_mean_accel_w = np.array([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z])
        

    def robot_state_callback(self, data):
        # robot state as [x, y, z, [w, x, y, z], vx, vy, vz]
        self.current_state = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z, data.pose.pose.orientation.w, data.pose.pose.orientation.x,
                                      data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z, ]).reshape(1, -1)

    def robot_pose_callback(self, msg):
        # robot state as [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        self.current_time = rospy.get_time()
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.velocity_estimation()
        if self.vel is not None:
            if msg.pose.orientation.w > 0:
                self.current_state = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, self.vel[0], self.vel[1], self.vel[2] ])
            elif msg.pose.orientation.w < 0:
                self.current_state = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        -msg.pose.orientation.w, -msg.pose.orientation.x, -msg.pose.orientation.y,
                        -msg.pose.orientation.z, self.vel[0], self.vel[1], self.vel[2] ])
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

    def trajectory_command_callback(self, data):
        temp_traj = data.traj
        if data.size != len(temp_traj):
            rospy.logerr('Some data are lost')
        else:
            self.trajectory_path = np.zeros((data.size, 10))
            # print("data.size = %d" % data.size) # data.size =1
            for i in range(data.size):
                # quaternion_ = self.rpy_to_quaternion(
                #     [temp_traj[i].roll, temp_traj[i].pitch, temp_traj[i].yaw])
                self.trajectory_path[i] = np.array([temp_traj[i].x,
                                                    temp_traj[i].y,
                                                    temp_traj[i].z,
                                                    temp_traj[i].q[0],
                                                    temp_traj[i].q[1],
                                                    temp_traj[i].q[2],
                                                    temp_traj[i].q[3],
                                                    # quaternion_[0],
                                                    # quaternion_[1],
                                                    # quaternion_[2],
                                                    # quaternion_[3],
                                                    temp_traj[i].vx,
                                                    temp_traj[i].vy,
                                                    temp_traj[i].vz,
                                                    ])

    def quadrotor_optimizer_setup(self, ):
        # Q_m_ = np.diag([10, 10, 10,
        #                 0.3, 0.3, 0.3, 0.3,
        #                 0.05, 0.05, 0.05,
        #                 ])  # position, q, v
        # P_m_ = np.diag([10, 10, 10,
        #                 0.05, 0.05, 0.05])  # only p and v
        # R_m_ = np.diag([5.0, 5.0, 5.0, 0.6])

        # parameters of c_generated_code/uav_q (safe_takeoff)
        Q_m_ = np.diag([10, 10, 10,
                    0.8, 0.8, 0.8, 0.8,
                    0.05, 0.05, 0.05,
                    ])  # position, q, v
        P_m_ = np.diag([
            10,  # x
            10,  # y
            20,  # z
            0.05,  # vx
            0.05,  # vy
            0.05  # vz
        ])
        R_m_ = np.diag([1.0, 1.0, 2.0, 0.6])

        nx = self.model.x.size()[0]
        self.nx = nx
        nu = self.model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = self.model.p.size()[0] if isinstance(
            self.model.p, ca.SX) else 0

        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = self.model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # cost type
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q_m_, R_m_)
        ocp.cost.W_e = P_m_

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.zeros((nx - 4, nx))
        # ocp.cost.Vx_e[:6, :6] = np.eye(6)
        ocp.cost.Vx_e[:3, :3] = np.eye(3)
        ocp.cost.Vx_e[-3:, -3:] = np.eye(3)

        # initial reference trajectory_ref
        x_ref = np.zeros(nx)
        x_ref[3] = 1.0
        x_ref_e = np.zeros(nx - 4)
        u_ref = np.zeros(nu)
        u_ref[-1] = self.g_
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref_e

        # Set constraints
        ocp.constraints.lbu = np.array([self.constraints.roll_rate_min, self.constraints.pitch_rate_min,
                                       self.constraints.yaw_rate_min, self.constraints.thrust_min])
        ocp.constraints.ubu = np.array([self.constraints.roll_rate_max, self.constraints.pitch_rate_max,
                                       self.constraints.yaw_rate_max, self.constraints.thrust_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # initial state
        ocp.constraints.x0 = x_ref

        # solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP'  # 'SQP_RTI'
        # ocp.solver_options.nlp_solver_max_iter = 400
        # parameters of c_generated_code/uav_q
        ocp.solver_options.nlp_solver_max_iter = 200

        # compile acados ocp
        # files are stored in .ros/
        json_file = os.path.join('./' + self.model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        if self.simulation_required:
            self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def mpc_estimation_loop(self,):
        t1 = time.time()
        if self.trajectory_path is not None and self.current_state is not None:
            current_trajectory = self.trajectory_path
            u_des = np.array([0.0, 0.0, 0.0, self.g_])
            self.solver.set(self.N, 'yref', np.concatenate(
                (current_trajectory[-1, :3], current_trajectory[-1, -3:])))
            for i in range(self.N):
                self.solver.set(i, 'yref', np.concatenate(
                    (current_trajectory[0], u_des)))

            self.solver.set(0, 'lbx', self.current_state.flatten())
            self.solver.set(0, 'ubx', self.current_state.flatten())
            # set parameter
            for i in range(self.N):
                self.solver.set(i, 'p', self.gp_mean_accel_w)

            status = self.solver.solve()

            if status != 0:
                rospy.logerr("MPC cannot find a proper solution.")
                # self.att_command.thrust = 0.5
                # self.att_command.thrust = 0.59
                self.att_command.thrust = self.control_offset
                self.att_command.body_rate.z = 0.0
                self.att_command.body_rate.x = 0.0
                self.att_command.body_rate.y = 0.0
                # only for debug
                # print(self.trajectory_path)
                # print("----")
                # print(self.current_state)
            else:
                mpc_x_ = self.solver.get(1, 'x')
                mpc_u_ = self.solver.get(0, 'u')
                self.att_command.body_rate.x = mpc_u_[0]
                self.att_command.body_rate.y = mpc_u_[1]
                self.att_command.body_rate.z = mpc_u_[2]
                # q330, circle_40s, control_offset=0.46
                self.att_command.thrust = mpc_u_[3] / self.g_ * self.control_offset 

                self.mpc_x_next_state.pose.pose.position.x = mpc_x_[0]
                self.mpc_x_next_state.pose.pose.position.y = mpc_x_[1]
                self.mpc_x_next_state.pose.pose.position.z = mpc_x_[2]
                self.mpc_x_next_state.pose.pose.orientation.w = mpc_x_[3]
                self.mpc_x_next_state.pose.pose.orientation.x = mpc_x_[4]
                self.mpc_x_next_state.pose.pose.orientation.y = mpc_x_[5]
                self.mpc_x_next_state.pose.pose.orientation.z = mpc_x_[6]
                self.mpc_x_next_state.twist.twist.linear.x = mpc_x_[7]
                self.mpc_x_next_state.twist.twist.linear.y = mpc_x_[8]
                self.mpc_x_next_state.twist.twist.linear.z = mpc_x_[9]
            # self.att_setpoint_pub.publish(self.att_command)
            self.mpc_x_next_pub.publish(self.mpc_x_next_state)

        else:
            if self.trajectory_path is None:
                rospy.loginfo("waiting trajectory")
            elif self.current_state is None:
                rospy.loginfo("waiting current state")
            else:
                rospy.loginfo("Unknown error")
        self.rate.sleep()
        # print(time.time()-t1)
        return True

    @staticmethod
    def quaternion_to_rpy(quaternion):
        q0, q1, q2, q3 = quaternion.w, quaternion.x, quaternion.y, quaternion.z
        roll_ = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        pitch_ = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw_ = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        return roll_, pitch_, yaw_

    @staticmethod
    def rpy_to_quaternion(rqy):
        roll_, pitch_, yaw_ = rqy
        cy = np.cos(yaw_ * 0.5)
        sy = np.sin(yaw_ * 0.5)
        cp = np.cos(pitch_ * 0.5)
        sp = np.sin(pitch_ * 0.5)
        cr = np.cos(roll_ * 0.5)
        sr = np.sin(roll_ * 0.5)

        w_ = cr * cp * cy + sr * sp * sy
        x_ = sr * cp * cy - cr * sp * sy
        y_ = cr * sp * cy + sr * cp * sy
        z_ = cr * cp * sy - sr * sp * cy

        return np.array([w_, x_, y_, z_])

    def state_server(self, req):
        return SetBoolResponse(True, 'MPC is ready')

    def send_command(self,):
        rate = rospy.Rate(100)  # Hz
        self.att_command.header = Header()

        while not rospy.is_shutdown():
            # t2 = time.time()
            command_ = self.att_command
            self.att_command.header.stamp = rospy.Time.now()
            self.att_setpoint_pub.publish(command_)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
            # print("publsich loop takes {} seconds".format(time.time() - t2))


if __name__ == '__main__':
    rospy.init_node('offboard_mpc_controller')
    quad_rotor_model = QuadRotorModel()
    try:
        mpc_obj = MPC_controller(quad_model=quad_rotor_model.model,
                                 quad_constraints=quad_rotor_model.constraints,
                                 t_horizon=2.,
                                 n_nodes=20
                                 )
        mpc_model_is_ready = True
    except ImportError:
        rospy.logerr('Cannot find any MPC library, Stop the node')
        mpc_model_is_ready = False
        mpc_obj = None

    while not rospy.is_shutdown() and mpc_model_is_ready:
        if not mpc_obj.mpc_estimation_loop():
            rospy.logerr("MPC estimation failed")

    print('MPC controller is shutdown')
