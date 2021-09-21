#!/usr/bin/env python
# coding=utf-8

'''
Date: 01.09.2021
Author: Yinfeng Long
last_edit: 19.09.2021
usage
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz']
'''

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from math import floor

class GpMean(object):
    def __init__(self, x_train_idx, y_train_idx, gp_model_file_path, npz_name):
        self.file_path = gp_model_file_path
        self.npz_name = npz_name

        self.train_x = None
        self.train_y = None
        self.load_data(x_train_idx, y_train_idx)
        self.model_to_predict = None
        self.likelihood_pred = None
        self.load_model(x_train_idx)
    
    def load_data(self, x_train_idx, y_train_idx):
        gp_train = np.load(self.file_path + '/' + self.npz_name)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        train_x_ori = (gp_train[x_train_idx]).flatten()
        train_y_ori = (gp_train[y_train_idx]).flatten()

        self.train_x = torch.from_numpy( train_x_ori[:7000] )
        self.train_y = torch.from_numpy( train_y_ori[:7000])

        # numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
        print("x_train shape", self.train_x.shape)
        print("y_train shape", self.train_y.shape)
        self.train_x = (self.train_x).float().to(device)
        self.train_y = (self.train_y).float().to(device)
    
    def load_model(self, x_train_idx):
        target_device = 'cuda:0'

        self.likelihood_pred = gpytorch.likelihoods.GaussianLikelihood()
        self.model_to_predict = ExactGPModel(self.train_x, self.train_y, self.likelihood_pred).to(target_device)

        if torch.cuda.is_available():
            model_state_dict = torch.load( self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth')
            likelihood_state_dict = torch.load(self.file_path + '/train_pre_model/likelihood_state_' + x_train_idx +'.pth')
        else:
            model_state_dict = torch.load(self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth', map_location=target_device)
            likelihood_state_dict = torch.load(self.file_path + '/train_pre_model/likelihood_state_' + x_train_idx+'.pth', map_location=target_device)
        self.model_to_predict.load_state_dict(model_state_dict)
        self.likelihood_pred.load_state_dict(likelihood_state_dict)

        self.model_to_predict.eval()
        self.likelihood_pred.eval()

    def predict_mean(self, test_point):
        target_device = 'cuda:0'

        test_x = torch.tensor(test_point).float().to(target_device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood_pred(self.model_to_predict(test_x))
            return observed_pred.mean.cpu().numpy()

##############################
# GpyTorch #
##############################
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
        


#################################
# GpyTorch using combined datas#
#################################
class GpMeanCombine(object):
    def __init__(self, x_train_idx, gp_model_file_path, npz_name):
        self.file_path = gp_model_file_path
        self.npz_name = npz_name
        
        self.train_x = None
        self.load_data( x_train_idx )
        self.model_to_predict = None
        self.load_model( x_train_idx)
    
    def load_data(self, x_train_idx ):
        gp_train = np.load(self.file_path + '/' + self.npz_name)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
        X = torch.from_numpy( (gp_train[x_train_idx]).flatten() )

        # 80% datas for training, 20% datas for testing
        train_n = int(floor(0.8 * len(X)))
        self.train_x = X[:train_n].contiguous()
        self.train_x = (self.train_x).float().to(device)
    
    def load_model(self, x_train_idx):
        target_device = 'cuda:0'

        inducing_points = self.train_x[:500]
        self.model_to_predict = GPModel(inducing_points=inducing_points)

        if torch.cuda.is_available():
            model_state_dict = torch.load( self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth')
        else:
            model_state_dict = torch.load(self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth', map_location=target_device)
        self.model_to_predict.load_state_dict(model_state_dict)
        self.model_to_predict = self.model_to_predict.to(target_device)

        self.model_to_predict.eval()

    def predict_mean(self, test_point):
        target_device = 'cuda:0'

        test_x = torch.tensor(test_point).float().to(target_device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model_to_predict( test_x )
            return observed_pred.mean.cpu().numpy()

##############################
# GpyTorch #
##############################
class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
