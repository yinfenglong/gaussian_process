#!/usr/bin/env python
# coding=utf-8

'''
Date: 19.09.2021
Author: Yinfeng Long
usage
    python3 gpr_GPyTorch_train.py q330/20210913_3_20_random
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz']
'''

import numpy as np
import torch
import gpytorch
import time
import sys

class GpTrain(object):
    def __init__(self, x_train_idx, y_train_idx, gp_model_file_path, npz_name ):
        self.file_path = gp_model_file_path
        self.npz_name = npz_name

        self.train_x = None 
        self.train_y = None
        # self.train_y_range = None
        self.load_data(x_train_idx, y_train_idx )

        self.observed_pred = None
        self.model_to_predict = self.train_model( x_train_idx)
    
    def load_data(self, x_train_idx, y_train_idx ):
        gp_train = np.load(self.file_path + '/' + self.npz_name)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
        train_x_ori = (gp_train[x_train_idx]).flatten()
        train_y_ori = (gp_train[y_train_idx]).flatten()

        self.train_x = torch.from_numpy( train_x_ori[:7000] )
        self.train_y = torch.from_numpy( train_y_ori[:7000] )
        print("dimension of x after prune:", np.array(self.train_x).shape)
        print("dimension of y after prune:", np.array(self.train_y).shape)

        self.train_x = (self.train_x).float().to(device)
        self.train_y = (self.train_y).float().to(device)

    def train_model(self, x_train_idx):
        device = 'cuda:0'
        time_1 = time.time()
        # device = torch.device("cuda")
        """
        # ### Prior theta Parameters ### #
        l_scale = 1 #0.1
        sigma_f = 0.5
        sigma_n = 0.01
        """
        hypers = {
            'covar_module.base_kernel.lengthscale': torch.tensor(1).to(device),
            'covar_module.outputscale': torch.tensor(0.5).to(device),
            'likelihood.noise_covar.noise': torch.tensor(0.01).to(device),
        }

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(self.train_x, self.train_y, likelihood).to(device)
        model.initialize(**hypers)
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 10
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

        time_2 = time.time()
        print("training time is: ", (time_2 - time_1))
        # torch.save(model.state_dict(), './model_state.pth')
        torch.save(model.state_dict(), './' +  sys.argv[1] + '/train_pre_model/model_state_' + x_train_idx +'.pth')
        likelihood_state_dict = likelihood.state_dict()
        # torch.save(likelihood_state_dict, './likelihood_state.pth')
        torch.save(likelihood_state_dict, './' + sys.argv[1] + '/train_pre_model/likelihood_state_' + x_train_idx +'.pth')

##############################
# GpyTorch #
##############################
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == '__main__':
    # ### From .npz load datas for gp training### #
    file_path = './' + sys.argv[1]
    npz_name = 'datas_for_gp_y.npz'
    gp_train = np.load( file_path + '/' + npz_name)

    x_idx_list = [i for i in gp_train.keys()][:6]
    y_idx_list = [i for i in gp_train.keys()][6:]
    for i in range(len(x_idx_list)):
        x_train_idx = x_idx_list[i]
        y_train_idx = y_idx_list[i]
        print("***************************")
        print("x_train_idx: {}".format(x_train_idx) )
        print("y_train_idx: {}".format(y_train_idx) )

        gpMPC = GpTrain(x_train_idx, y_train_idx, file_path, npz_name)
