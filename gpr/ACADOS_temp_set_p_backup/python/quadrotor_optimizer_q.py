#!/usr/bin/env python
# coding: utf-8
'''
Author: Wei Luo
Date: 2021-03-14 22:01:33
LastEditors: Yinfeng LOng
LastEditTime: 2021-08-27
Note: Note
'''

import os
import sys
# from utils.utils import safe_mkdir_recursive
from quadrotor_model_q import QuadRotorModel
from quadrotor_model_q_set_p import QuadRotorSetPModel 
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca
import scipy.linalg
import numpy as np
import time

import matplotlib.pyplot as plt

class QuadOptimizer:
    def __init__(self, quad_model, quad_model_p, quad_constraints, t_horizon, n_nodes ):
        model_ori = quad_model
        model_p = quad_model_p
        constraints = quad_constraints
        self.g_ = 9.81
        self.T = t_horizon
        self.N = n_nodes

        # Ensure current working directory is current folder
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        # safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        nx = model_ori.x.size()[0]
        self.nx = nx
        nu = model_ori.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = model_ori.p.size()[0] if isinstance(model_ori.p, ca.SX) else 0
        n_params_p = model_p.p.size()[0] if isinstance(model_p.p, ca.SX) else 0

        # create OCP
        ocp_ori = self.create_ocp(model_ori, constraints, nx, nu, ny, n_params, t_horizon)
        ocp_p = self.create_ocp(model_p, constraints, nx, nu, ny, n_params_p, t_horizon)
        
        # compile acados ocp
        json_file_p = os.path.join('./'+model_p.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp_p, json_file=json_file_p)

        # compile sim solver (without parameter)
        json_file_ori = os.path.join('./'+model_ori.name+'_acados_ocp.json')
        self.solver_p = AcadosOcpSolver(ocp_ori, json_file=json_file_ori)
        self.integrator = AcadosSimSolver(ocp_ori, json_file=json_file_ori)
        print("compile model without parameter")

    def create_ocp(self, model, constraints, nx, nu, ny, n_params, t_horizon):
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)
        
        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        # # AcadosSimSolver need ocp.dims.nx
        # ocp.dims.nx = self.nx
        ocp.dims.N = self.N
        ocp.solver_options.tf = t_horizon

        # initialize parameters
        ocp.dims.np = n_params
        print("n_params", ocp.dims.np)
        ocp.parameter_values = np.zeros(n_params)

        Q_m_ = np.diag([10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # position, q, velocity

        P_m_ = np.diag([10.0, 10.0, 10.0, 0.05, 0.05, 0.05])   # only p and v
        
        R_m_ = np.diag([5.0, 5.0, 5.0, 0.1]) # w, t

        # Ensure current working directory is current folder
        # cost type
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q_m_, R_m_)
        ocp.cost.W_e = P_m_

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.zeros((nx-4, nx))
        ocp.cost.Vx_e[:3, :3] = np.eye(3)
        ocp.cost.Vx_e[-3:, -3:] = np.eye(3)

        # initial reference trajectory_ref
        x_ref = np.zeros(nx)
        x_ref[3] = 1.0
        x_ref_e = np.zeros(nx-4)
        u_ref = np.zeros(nu)
        u_ref[-1] = self.g_
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref_e

        # Set constraints
        ocp.constraints.lbu = np.array([constraints.roll_rate_min, constraints.pitch_rate_min, constraints.yaw_rate_min, constraints.thrust_min])
        ocp.constraints.ubu = np.array([constraints.roll_rate_max, constraints.pitch_rate_max, constraints.yaw_rate_max, constraints.thrust_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # initial state
        ocp.constraints.x0 = x_ref

        # solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP' # 'SQP_RTI'

        return ocp

    def trajectory_generator(self, iter, current_state, current_trajectory):
        next_trajectories = current_trajectory[1:, :]
        next_trajectories = np.concatenate((next_trajectories,
        np.array([np.cos((iter)/30), np.sin((iter)/30), 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)))

        return next_trajectories

    def simulation(self, x0, user_trajectory=None):
        sim_time = 300 # s
        dt = 0.1 # s
        simX = np.zeros((int(sim_time/dt+1), self.nx))
        simD = np.zeros((int(sim_time/dt+1), self.nx))
        simU = np.zeros((int(sim_time/dt), self.nu))
        
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)
        simD[0, :] = x0.reshape(1, -1)
        if user_trajectory is None:
            init_trajectory = np.array(
                    [
                    [0.1, 0.0, 0.67, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.69, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.69, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.73, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.73, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.83, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.83, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.85, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.88, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.91, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.93, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.97, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.7, 0.0, 0.99, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.8, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.9, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]) # 20x10
        else:
            init_trajectory = user_trajectory

        mpc_iter = 0
        current_trajectory = init_trajectory.copy()
        u_des = np.array([0.0, 0.0, 0.0, self.g_])

        # current_trajectory = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t1 = time.time()
        while(mpc_iter <sim_time/dt and mpc_iter < 208):
            # define cost constraints
            self.solver.set(self.N, 'yref', np.concatenate((current_trajectory[-1, :3], current_trajectory[-1, -3:])))
            # self.solver.set(0, 'yref', np.concatenate([x_current, u_des]))
            for i in range(self.N):
                self.solver.set(i, 'yref', np.concatenate([current_trajectory[i], u_des]))

            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            # set parameter
            for j in range(0, self.N):
                self.solver.set(j, 'p', np.array([0.0, 0.0, -0.286]))
                # self.acados_ocp_solver[use_model].set(j, 'p', np.array([0.0] * (len(gp_state) + 1)))

            status = self.solver.solve()
            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
            simU[mpc_iter, :] = self.solver.get(0, 'u')
            simU[mpc_iter, 3] = 0.971*simU[mpc_iter, 3] 

            # print(current_trajectory)
            # print('-----')
            # for i in range(self.N):
            #     print(self.solver.get(i, 'x'))
            # print('-----')
            # print(simU[mpc_iter, :])
            # simulated system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[mpc_iter, :])
            status = self.integrator.solve()
            if status != 0:
                raise Exception('acados integrator returned status {}. Exiting.'.format(status))
            # update
            x_current = self.integrator.get('x')
            # print(x_current)
            # print('-----')
            # print('x des {}'.format(current_trajectory[0]))
            simX[mpc_iter+1, :] = x_current
            simD[mpc_iter+1, :] = current_trajectory[0]
            # get new trajectory_ref
            current_trajectory = self.trajectory_generator(mpc_iter, x_current, current_trajectory)
            # next loop
            mpc_iter += 1

        print('average time is {}'.format((time.time()-t1)/mpc_iter))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(simX[:mpc_iter, 0], simX[:mpc_iter, 1], simX[:mpc_iter, 2], 'b-.')
        ax.plot(simD[:mpc_iter, 0], simD[:mpc_iter, 1], simD[:mpc_iter, 2], 'r')
        ax.legend(['trajectory with parameter:0,0,-1.92', 'original trajectory'])
        manger = plt.get_current_fig_manager()
        manger.window.showMaximized()
        fig = plt.gcf()
        plt.show()
        fig.savefig('../../../thesis_figures/svg/' + 'xyz.svg', format='svg', dpi=800 )

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(range(mpc_iter), simX[:mpc_iter, 2], 'b-.')
        ax.plot(range(mpc_iter), simD[:mpc_iter, 2], 'r')
        ax.legend(['z with parameter: 0,0,-1.92', 'z'])
        # plt.title("1.659->1.709")
        manger = plt.get_current_fig_manager()
        manger.window.showMaximized()
        fig = plt.gcf()
        plt.show()
        fig.savefig('../../../thesis_figures/svg/' + 'z.svg', format='svg', dpi=800 )

if __name__ == '__main__':
    quad_model = QuadRotorModel()
    quad_model_p = QuadRotorSetPModel()
    opt = QuadOptimizer(quad_model.model, quad_model_p.model, quad_model.constraints, t_horizon=3., n_nodes=10)
    x_init = np.zeros(10)
    x_init[3] = 1
    opt.simulation(x0=x_init)
