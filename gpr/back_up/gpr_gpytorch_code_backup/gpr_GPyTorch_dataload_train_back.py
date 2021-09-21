#!/usr/bin/env python
# coding=utf-8

'''
Date: 18.09.2021
Author: Yinfeng Long
usage
    python3 gpr_GPyTorch_dataload_train.py q330 

    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz'] 
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz'] 

'''

import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import time
from math import floor
from tqdm.notebook import tqdm
import tqdm

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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

time_1 = time.time()
# ### From .npz load datas for gp training### #
gp_train = np.load( './' + sys.argv[1] + '/combined_q330.npz' )
x_idx_list = [i for i in gp_train.keys()][:6]
y_idx_list = [i for i in gp_train.keys()][6:]

for i in range(len(x_idx_list)):
    x_train_idx = x_idx_list[i]
    y_train_idx = y_idx_list[i]
    print("***************************")
    print("x_train_idx: {}".format(x_train_idx) )
    print("y_train_idx: {}".format(y_train_idx) )
    # numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
    X = torch.from_numpy( (gp_train[x_train_idx]).flatten() )
    y = torch.from_numpy( (gp_train[y_train_idx]).flatten() )
    print("dimension of x:", np.array(X).shape)
    print("dimension of y:", np.array(y).shape)

    # 80% datas for training, 20% datas for testing
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n].contiguous()
    train_y = y[:train_n].contiguous()

    train_x = (train_x).float().to(device)
    train_y = (train_y).float().to(device)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
 
    """
    # ### Prior theta Parameters ### #
    l_scale = 1 #0.1
    sigma_f = 0.5
    sigma_n = 0.01
    """
    # hypers = {
    #     'covar_module.base_kernel.lengthscale': torch.tensor(1).to(device),
    #     'covar_module.outputscale': torch.tensor(0.5).to(device),
    #     'likelihood.noise_covar.noise': torch.tensor(0.01).to(device),
    # }

    # initialize likelihood and model
    inducing_points = train_x[:500]
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(inducing_points=inducing_points).to(device)
    # model.initialize(**hypers)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    num_epochs = 4
    epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    # training_iter = 10
    # for i in range(training_iter):
    #     # Zero gradients from previous iteration
    #     optimizer.zero_grad()
    #     # Output from model
    #     output = model(train_x)
    #     # Calc loss and backprop gradients
    #     loss = -mll(output, train_y)
    #     loss.backward()
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss.item(),
    #         model.covar_module.base_kernel.lengthscale.item(),
    #         model.likelihood.noise.item()
    #     ))
    #     optimizer.step()

    time_2 = time.time()
    print("training time is: ", (time_2 - time_1))
    # torch.save(model.state_dict(), './model_state.pth')
    torch.save(model.state_dict(), './' +  sys.argv[1] + '/train_pre_model/model_state_' + x_train_idx +'.pth')
    likelihood_state_dict = likelihood.state_dict()
    # torch.save(likelihood_state_dict, './likelihood_state.pth')
    torch.save(likelihood_state_dict, './' + sys.argv[1] + '/train_pre_model/likelihood_state_' + x_train_idx +'.pth')



















