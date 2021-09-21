#!/usr/bin/env python
# coding=utf-8

'''
Date: 15.08.2021
Author: Yinfeng Long
usage
    python3 gpr_GPyTorch_predict.py file_name.npz x_train_idx y_train_idx
    x_train_idx = ['x', 'y', 'z', 'vx', 'vy', 'vz'] ['arr_0']
    y_train_idx = ['y_x', 'y_y', 'y_z', 'y_vx', 'y_vy', 'y_vz'] ['arr_1']

    python3 gpr_GPyTorch_predict.py q330 vz y_vz
'''

import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from matplotlib import pyplot as plt
import time
from math import floor

##############################
# prediction #
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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

# ### From .npz load datas for gp training### #
gp_train = np.load( './' + sys.argv[1] + '/combined_q330.npz' )
x_idx_list = [i for i in gp_train.keys()][:6]
y_idx_list = [i for i in gp_train.keys()][6:]

for i in range(len(x_idx_list)):
# for i in range(1):
    x_train_idx = x_idx_list[i]
    y_train_idx = y_idx_list[i]
    print("***************************")
    print("x_train_idx: {}".format(x_train_idx) )
    print("y_train_idx: {}".format(y_train_idx) )
    # numpy into one dimension, then create a Tensor form from numpy (=torch.linspace)
    X = (gp_train[x_train_idx]).flatten()
    y = (gp_train[y_train_idx]).flatten()
    train_y_range = np.max(y) - np.min(y)
    X = torch.from_numpy( X )
    y = torch.from_numpy( y )
    print("dimension of x:", np.array(X).shape)
    print("dimension of y:", np.array(y).shape)

    # 80% datas for training, 20% datas for testing
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:].contiguous()
    test_y = y[train_n:].contiguous()

    train_x = (train_x).float().to(device)
    train_y = (train_y).float().to(device)
    test_x = (test_x).float().to(device)
    test_y = (test_y).float().to(device)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # if not torch.cuda.is_available():
    #     device = torch.device("cpu")
    #     target_device = 'cpu'
    # else:
    #     device = torch.device("cuda")
    #     target_device = 'cuda:0'
    target_device = 'cuda:0'

    inducing_points = train_x[:500]
    likelihood_pred = gpytorch.likelihoods.GaussianLikelihood()
    model_to_predict = GPModel(inducing_points=inducing_points)

    if torch.cuda.is_available():
        model_state_dict = torch.load('./' + sys.argv[1] + '/train_pre_model/model_state_' + x_train_idx +'.pth')
        likelihood_state_dict = torch.load('./' + sys.argv[1] + '/train_pre_model/likelihood_state_' + x_train_idx +'.pth')
    else:
        model_state_dict = torch.load('./' + sys.argv[1] + '/train_pre_model/model_state_' + x_train_idx +'.pth', map_location=target_device)
        likelihood_state_dict = torch.load('./' + sys.argv[1] +'/train_pre_model/likelihood_state_' + x_train_idx +'.pth', map_location=target_device)
    model_to_predict.load_state_dict(model_state_dict)
    likelihood_pred.load_state_dict(likelihood_state_dict)
    model_to_predict = model_to_predict.to(target_device)
    likelihood_pred = likelihood_pred.to(target_device)

    model_to_predict.eval()
    likelihood_pred.eval()

    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            test_x = x_batch
            observed_pred = model_to_predict( x_batch.to(target_device) )
            means = torch.cat([means, observed_pred.mean.cpu()])
    means = means[1:]
    print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))

    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     # test_x = train_x
    #     t1 = time.time()
    #     # t1 = datetime.datetime.now()
    #     observed_pred = likelihood_pred(model_to_predict(test_x.to(target_device)))
    #     t2 = time.time()
    #     # t2 = datetime.datetime.now()
    #     print("predict time of GPyTorch (predict 100 new numbers):",
    #         (t2 - t1))
        
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        train_x_cpu = train_x.cpu()
        train_y_cpu = train_y.cpu()
        if target_device == 'cuda:0':
            test_x_cpu = test_x.cpu()
        else:
            test_x_cpu = test_x
        # Plot training data as black stars
        ax.plot(train_x_cpu.numpy(), train_y_cpu.numpy(), 'k*', alpha=0.5)
        ax.plot(test_x_cpu.numpy(), observed_pred.mean.cpu().numpy(), 'r*')

        print("observed_pred.mean.numpy(): ", observed_pred.mean.cpu().numpy())
        print("observed_pred.mean.numpy().shape: ", observed_pred.mean.cpu().numpy().shape )
        ax.errorbar(test_x_cpu.numpy(), observed_pred.mean.cpu().numpy(), yerr=np.concatenate(((-lower.cpu().numpy() +
                    observed_pred.mean.cpu().numpy()).reshape(1, -1), (upper.cpu().numpy() - observed_pred.mean.cpu().numpy()).reshape(1, -1)), axis=0), ecolor='b', elinewidth=2, capsize=4, fmt=' ')
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        if train_y_range < 2:
            maloc = 0.05 
            miloc = 0.05
        else:
            # maloc = float( '%.1f'%(train_y_range/30))
            # miloc = maloc / 2
            maloc = 1 
            miloc = 0.5
        print("maloc:", maloc)
        print("train_y_range:", train_y_range)
        ax.yaxis.set_major_locator( plt.MultipleLocator(maloc) )
        ax.yaxis.set_minor_locator( plt.MultipleLocator(miloc) )
        ax.grid(axis='y', which='major', color='darkorange', alpha=1)
        ax.grid(axis='y', which='minor', color='darkorange', alpha=0.5)

        plt.title( sys.argv[1] + '/' + x_train_idx )
        manger = plt.get_current_fig_manager()
        manger.window.showMaximized()
        fig = plt.gcf()
        plt.show()
        fig.savefig( './' + sys.argv[1] + '/figures/'+ x_train_idx + '.png' )