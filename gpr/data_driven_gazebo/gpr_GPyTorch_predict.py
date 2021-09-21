#!/usr/bin/env python
# coding=utf-8

'''
Date: 15.08.2021
Author: Yinfeng Long
usage
    python3 gpr_GPyTorch_predict.py file_name.npz x_train_idx y_train_idx
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz']
'''

import sys
import numpy as np
import joblib
import torch
import gpytorch
from matplotlib import pyplot as plt
import datetime
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

# # ### From .npz load datas for gp training### #
gp_train = np.load(sys.argv[1])
x_train_idx = sys.argv[2]
y_train_idx = sys.argv[3]
train_x_ori = (gp_train[x_train_idx]).flatten()
train_y_ori = (gp_train[y_train_idx]).flatten()

if x_train_idx in ['vx', 'vy', 'vz']:
    train_x = []
    train_y = []
    for i in range(train_x_ori.shape[0]):
        if abs(train_x_ori[i])>0.01:
            train_x.append( train_x_ori[i])
            train_y.append( train_y_ori[i])
    train_x = torch.from_numpy( np.array(train_x) )
    train_y = torch.from_numpy( np.array(train_y) ) 
    
else:
    train_x = torch.from_numpy( train_x_ori[:7000] )
    train_y = torch.from_numpy( train_y_ori[:7000])

# numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
print("x_train shape", train_x.shape)
print("y_train shape", train_y.shape)

train_x = (train_x).float().to(device)
train_y = (train_y).float().to(device)

##############################
# prediction #
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

# if not torch.cuda.is_available():
#     device = torch.device("cpu")
#     target_device = 'cpu'
# else:
#     device = torch.device("cuda")
#     target_device = 'cuda:0'

target_device = 'cuda:0'

likelihood_pred = gpytorch.likelihoods.GaussianLikelihood()
model_to_predict = ExactGPModel(train_x, train_y, likelihood_pred).to(target_device)

if torch.cuda.is_available():
    model_state_dict = torch.load('./train_pre_model/model_state_' + sys.argv[2] +'.pth')
    likelihood_state_dict = torch.load('./train_pre_model/likelihood_state' + sys.argv[2] +'.pth')
else:
    model_state_dict = torch.load('./train_pre_model/model_state_' + sys.argv[2] +'.pth', map_location=target_device)
    likelihood_state_dict = torch.load('./train_pre_model/likelihood_state' + sys.argv[2] +'.pth', map_location=target_device)
model_to_predict.load_state_dict(model_state_dict)
likelihood_pred.load_state_dict(likelihood_state_dict)

model_to_predict.eval()
likelihood_pred.eval()

# Test points are regularly spaced along [-16,16]
# Make predictions by feeding model through likelihood
# if sys.argv[2] == 'z':
#     test_x = torch.linspace(0.3, 1.3, 100, dtype=torch.float).to(target_device)
# elif sys.argv[2] == 'x' or sys.argv[2] == 'y':
#     test_x = torch.linspace(-0.7, 0.8, 100, dtype=torch.float).to(target_device)
# else:
#     test_x = torch.linspace(-0.6, 0.6, 100, dtype=torch.float).to(target_device)

test_x = torch.rand(31) * (8) 
# test_x = train_x.to(target_device)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = train_x
    t1 = time.time()
    # t1 = datetime.datetime.now()
    # for i in range(10):
    observed_pred = likelihood_pred(model_to_predict(test_x.to(target_device)))
    t2 = time.time()
    # t2 = datetime.datetime.now()
    print("predict time of GPyTorch (predict 100 new numbers):",
          (t2 - t1))
    
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    train_x_cpu = train_x.cpu()
    train_y_cpu = train_y.cpu()
    if target_device == 'cuda:0':
        test_x_cpu = test_x.cpu()
    else:
        test_x_cpu = test_x
    # Plot training data as black stars
    ax.plot(train_x_cpu.numpy(), train_y_cpu.numpy(), 'k*', alpha=0.3)
    # print("test_x: ", test_x.numpy())
    # Plot predictive means as blue line
    ax.plot(test_x_cpu.numpy(), observed_pred.mean.cpu().numpy(), 'm*')
    # ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'bs')
    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x_cpu.numpy(), lower.cpu().numpy(),
    #                 upper.cpu().numpy(), alpha=0.5)
    # if sys.argv[2] == 'z':
    #     # ax.set_ylim([-2, 2])
    #     pass
    # elif sys.argv[2] == 'x' or sys.argv[2] == 'y':
    #     ax.set_ylim( [-0.5, 0.5])
    # else:
    #     ax.set_ylim( [-4,4])
    ax.errorbar(test_x_cpu.numpy(), observed_pred.mean.cpu().numpy(), yerr=np.concatenate(((-lower.cpu().numpy() +
                observed_pred.mean.cpu().numpy()).reshape(1, -1), (upper.cpu().numpy() - observed_pred.mean.cpu().numpy()).reshape(1, -1)), axis=0), ecolor='r', elinewidth=2, capsize=4, fmt=' ')
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    print("observed_pred.mean.numpy(): ", observed_pred.mean.cpu().numpy())
    print("observed_pred.mean.numpy().shape: ", observed_pred.mean.cpu().numpy().shape )
plt.show()



        
