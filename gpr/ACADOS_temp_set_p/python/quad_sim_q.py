#!/usr/bin/env python
# coding=UTF-8
'''
Author: Wei Luo
Date: 2021-09-28 18:27:41
LastEditors: Wei Luo
LastEditTime: 2021-11-15 21:51:28
Note: Note
'''
from quadrotor_model_q import QuadRotorModel
from quadrotor_optimizer_q import QuadOptimizer
from quadrotor_model_q_set_p import QuadRotorSetPModel
import numpy as np

if __name__ == '__main__':
    # quad_model = QuadRotorModel()
<<<<<<< HEAD
=======
    # quad_model = QuadRotorSetPModel(mass_offset=0.29)
    # quad_model_p = QuadRotorSetPModel(mass_offset=0.29)
>>>>>>> a523b6c7451162f569c3fa5702f38465010b7bc1
    quad_model = QuadRotorModel()
    quad_model_p = QuadRotorModel()
    opt = QuadOptimizer(quad_model.model,
                        quad_model_p.model,
                        quad_model.constraints,
                        t_horizon=2.,
                        n_nodes=20)
    x_init = np.zeros(10)
    x_init[3] = 1
    x_init = np.array([
        1., 0., 0.66277079, 0.99997466, 0.00473137, -0.004642, -0.00259799,
        -0.02888979, -0.03484652, -0.12514812
    ])
    traj_init = np.tile(
        np.array([
            1., 0.0, 1., 0.99998597, 0., 0., -0.00529701, 0.78539816,
            -0.02398552, -0.68992918
        ]), (20, 1))
    opt.simulation(x0=x_init, user_trajectory=traj_init)
