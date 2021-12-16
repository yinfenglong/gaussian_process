
from quadrotor_model_q import QuadRotorModel
from quadrotor_optimizer_q import QuadOptimizer 
from quadrotor_model_q_set_p import QuadRotorSetPModel 
import numpy as np


if __name__ == '__main__':
    quad_model = QuadRotorModel()
    quad_model_p = QuadRotorSetPModel()
    opt = QuadOptimizer(quad_model.model, quad_model_p.model, quad_model.constraints, t_horizon=2., n_nodes=20)
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
    '''
    x_init = np.array([2.59003495, 0.58640395, 0.66277079, 0.99997466, 0.00473137,-0.004642, 
  -0.00259799,-0.02888979, -0.03484652, -0.12514812])
    traj_init = np.tile(np.array([ 2.6, 0.6, 1., 0.99998597, 0.,0. ,-0.00529701, 0.78539816, -0.02398552, -0.68992918]), (20, 1))
    '''
    opt.simulation(x0=x_init, user_trajectory=traj_init)

