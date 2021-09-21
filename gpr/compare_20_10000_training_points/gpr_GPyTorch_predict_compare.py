#!/usr/bin/env python
# coding=utf-8

'''
Date: 25.08.2021
Author: Yinfeng Long
usage
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz'] ['arr_0']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz'] ['arr_1']
'''

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

def load_data( x_train_idx, y_train_idx, file_path ):
    gp_train = np.load(file_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
    return train_x, train_y

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

def load_model(train_x, train_y, model_state, likelihood_state):
    target_device = 'cuda:0'

    likelihood_pred = gpytorch.likelihoods.GaussianLikelihood()
    model_to_predict = ExactGPModel(train_x, train_y, likelihood_pred).to(target_device)

    if torch.cuda.is_available():
        model_state_dict = torch.load( model_state)
        likelihood_state_dict = torch.load(likelihood_state)
    else:
        model_state_dict = torch.load(model_state, map_location=target_device)
        likelihood_state_dict = torch.load(likelihood_state, map_location=target_device)
    model_to_predict.load_state_dict(model_state_dict)
    likelihood_pred.load_state_dict(likelihood_state_dict)

    model_to_predict.eval()
    likelihood_pred.eval()
    return model_to_predict, likelihood_pred

def predict_mean( file_path, x_train_idx, y_train_idx, model_state, likelihood_state ):
    # file_path = '/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/GPR'
    train_x, train_y = load_data(x_train_idx, y_train_idx, file_path)
    model_to_predict, likelihood_pred = load_model(train_x, train_y, model_state, likelihood_state )
    
    target_device = 'cuda:0'

    # test_x = torch.tensor(test_point).float().to(target_device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.rand(31) * (12 + 11) - 11
        observed_pred = likelihood_pred(model_to_predict(test_x.to(target_device)))
        return train_x, train_y, observed_pred, test_x
        # return observed_pred.mean.cpu().numpy()

if __name__ == '__main__':
    train_x_1, train_y_1, observed_pred_1, test_x_1 = predict_mean( \
        './cluster_data_for_gp_z.npz', 'arr_0', 'arr_1','./train_pre_model/model_state_7000.pth', './train_pre_model/likelihood_state_7000.pth' )
    train_x_2, train_y_2, observed_pred_2, test_x_2 = predict_mean( \
        './for_gp_data_z.npz', 'arr_0', 'arr_1', './train_pre_model/model_state_20.pth', './train_pre_model/likelihood_state_20.pth')

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower1, upper1 = observed_pred_1.confidence_region()
        lower2, upper2 = observed_pred_2.confidence_region()

        train_x_cpu_1 = train_x_1.cpu()
        train_y_cpu_1 = train_y_1.cpu()
        test_x_cpu_1 = test_x_1.cpu()
        train_x_cpu_2 = train_x_2.cpu()
        train_y_cpu_2 = train_y_2.cpu()
        test_x_cpu_2 = test_x_2.cpu()
        # Plot training data as black stars
        ax.plot(train_x_cpu_1.numpy(), train_y_cpu_1.numpy(), 'k*', alpha=0.1 )
        # Plot predictive means as blue line
        ax.plot(test_x_cpu_1.numpy(), observed_pred_1.mean.cpu().numpy(), 'g^')
        # print(test_x_cpu.numpy().shape)
        # print(observed_pred.mean.cpu().numpy().shape)
        # print(np.concatenate((lower.cpu().numpy().reshape(-1, 1),
        #       upper.cpu().numpy().reshape(-1, 1)), axis=1).shape)
        ax.errorbar(test_x_cpu_1.numpy(), observed_pred_1.mean.cpu().numpy(), yerr=np.concatenate(((-lower1.cpu().numpy() +
                    observed_pred_1.mean.cpu().numpy()).reshape(1, -1), (upper1.cpu().numpy() - observed_pred_1.mean.cpu().numpy()).reshape(1, -1)), axis=0), ecolor='b', elinewidth=2, capsize=4, fmt=' ')
        ax.plot(train_x_cpu_2.numpy(), train_y_cpu_2.numpy(), 'm*' )
        ax.plot(test_x_cpu_2.numpy(), observed_pred_2.mean.cpu().numpy(), 'r^')
        ax.errorbar(test_x_cpu_2.numpy(), observed_pred_2.mean.cpu().numpy(), yerr=np.concatenate(((-lower2.cpu().numpy() +
                    observed_pred_2.mean.cpu().numpy()).reshape(1, -1), (upper2.cpu().numpy() - observed_pred_2.mean.cpu().numpy()).reshape(1, -1)), axis=0), ecolor='orange', elinewidth=2, capsize=4, fmt=' ')
        # print(test_x_cpu.numpy())
        # print(observed_pred.mean.cpu().numpy())
        # print(lower.cpu().numpy())
        # print(upper.cpu().numpy())
        ax.legend(['Observed Data_7000', 'Mean_7000', 'Observed Data_20', 'Mean_20', 'Confidence_7000', 'Confidence_20'])
        # ax.legend(handles=[t1, m1, e1, t2, m2, e2], \
        #     labels=['Observed Data_20', 'Mean_20','Confidence_20', 'Observed Data_7000', 'Mean_7000', 'Confidence_7000'])
        plt.show()

