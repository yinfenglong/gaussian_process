#!/usr/bin/env python
# coding=utf-8

'''
Date: 19.09.2021
Author: Yinfeng Long
usage
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz']
'''

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from math import floor
import time
import sys
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import tqdm
import os.path

class GpTrainCombine(object):
    def __init__(self, x_train_idx, y_train_idx, gp_model_file_path, npz_name ):
        self.file_path = gp_model_file_path
        self.npz_name = npz_name

        self.train_x = None 
        self.train_y = None
        self.train_loader = None 
        self.train_y_range = None
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
        X = (gp_train[x_train_idx]).flatten()
        y = (gp_train[y_train_idx]).flatten()
        np.random.shuffle(X)
        np.random.shuffle(y)
        self.train_y_range = np.max(y) - np.min(y)
        X = torch.from_numpy( X )
        y = torch.from_numpy( y )
        print("dimension of x:", np.array(X).shape)
        print("dimension of y:", np.array(y).shape)

        # 80% datas for training, 20% datas for testing
        train_n = int(floor(0.8 * len(X)))
        train_x = X[:train_n].contiguous()
        train_y = y[:train_n].contiguous()

        self.train_x = (train_x).float().to(device)
        self.train_y = (train_y).float().to(device)
        train_dataset = TensorDataset(self.train_x, self.train_y)
        self.train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


    def train_model(self, x_train_idx):
        device = 'cuda:0'
        time_1 = time.time()
        # initialize likelihood and model
        inducing_points = self.train_x[:500]
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel(inducing_points=inducing_points).to(device)

        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=self.train_y.size(0))

        num_epochs = 4
        epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.notebook.tqdm(self.train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

        time_2 = time.time()
        print("training time is: ", (time_2 - time_1))
        # torch.save(model.state_dict(), './model_state.pth')
        model_path = './' +  sys.argv[1] + '/train_pre_model/'
        if not os.path.exists(model_path):
            os.makedirs( model_path )
        torch.save(model.state_dict(), model_path + 'model_state_' + x_train_idx +'.pth')
        # likelihood_state_dict = likelihood.state_dict()
        # torch.save(likelihood_state_dict, './likelihood_state.pth')
        # torch.save(likelihood_state_dict, './' + sys.argv[1] + '/train_pre_model/likelihood_state_' + x_train_idx +'.pth')

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

if __name__ == '__main__':
    # ### From .npz load datas for gp training### #
    file_path = './' + sys.argv[1]
    # npz_name = 'combined_q330.npz'
    npz_name = sys.argv[2] 
    gp_train = np.load( file_path + '/' + npz_name)

    x_idx_list = [i for i in gp_train.keys()][:6]
    y_idx_list = [i for i in gp_train.keys()][6:]
    for i in range(len(x_idx_list)):
        x_train_idx = x_idx_list[i]
        y_train_idx = y_idx_list[i]
        print("***************************")
        print("x_train_idx: {}".format(x_train_idx) )
        print("y_train_idx: {}".format(y_train_idx) )

        gpMPC = GpTrainCombine(x_train_idx, y_train_idx, file_path, npz_name)
