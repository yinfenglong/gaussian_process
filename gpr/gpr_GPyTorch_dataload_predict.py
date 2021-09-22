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
import os.path

class GpMeanCombine(object):
    def __init__(self, x_train_idx, y_train_idx, gp_model_file_path, npz_name ):
        self.file_path = gp_model_file_path
        self.npz_name = npz_name

        self.train_x = None 
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.test_loader = None 
        self.train_y_range = None
        self.load_data(x_train_idx, y_train_idx )

        self.observed_pred = None
        self.model_to_predict = self.load_model( x_train_idx)
    
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
        test_x = X[train_n:].contiguous()
        test_y = y[train_n:].contiguous()

        self.train_x = (train_x).float().to(device)
        self.train_y = (train_y).float().to(device)
        self.test_x = (test_x).float().to(device)
        self.test_y = (test_y).float().to(device)

        test_dataset = TensorDataset(test_x, test_y)
        self.test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    def load_model(self, x_train_idx):
        target_device = 'cuda:0'

        inducing_points = self.train_x[:500]
        model_to_predict = GPModel(inducing_points=inducing_points)

        if torch.cuda.is_available():
            model_state_dict = torch.load( self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth')
        else:
            model_state_dict = torch.load(self.file_path + '/train_pre_model/model_state_' + x_train_idx +'.pth', map_location=target_device)
        model_to_predict.load_state_dict(model_state_dict)
        model_to_predict = model_to_predict.to(target_device)

        model_to_predict.eval()
        return model_to_predict 

    def predict_test(self, ):
        target_device = 'cuda:0'

        means = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                self.test_x = x_batch
                self.observed_pred = self.model_to_predict( x_batch.float().to(target_device) )
                means = torch.cat([means, self.observed_pred.mean.cpu()])
        means = means[1:]
        print('Test MAE: {}'.format(torch.mean(torch.abs(means - self.test_y.cpu()))))

    def plot_predict_result(self,):
        target_device = 'cuda:0'
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)

            # Get upper and lower confidence bounds
            lower, upper = self.observed_pred.confidence_region()
            train_x_cpu = self.train_x.cpu()
            train_y_cpu = self.train_y.cpu()
            if target_device == 'cuda:0':
                test_x_cpu = self.test_x.cpu()
            else:
                test_x_cpu = self.test_x
            # Plot training data as black stars
            ax.plot(train_x_cpu.numpy(), train_y_cpu.numpy(), 'k*', alpha=0.5)
            observed_pred_np = self.observed_pred.mean.cpu().numpy()
            ax.plot(test_x_cpu.numpy(), observed_pred_np, 'r*')

            print("observed_pred.mean.numpy(): ", observed_pred_np)
            print("observed_pred.mean.numpy().shape: ", observed_pred_np.shape )
            ax.errorbar(test_x_cpu.numpy(), observed_pred_np, yerr=np.concatenate(((-lower.cpu().numpy() +
                        observed_pred_np).reshape(1, -1), (upper.cpu().numpy() - observed_pred_np).reshape(1, -1)), axis=0), ecolor='b', elinewidth=2, capsize=4, fmt=' ')
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            
            # compare test_x predicted value with test_y

            if self.train_y_range < 2:
                maloc = 0.05 
                miloc = 0.05
            else:
                # maloc = float( '%.1f'%(train_y_range/30))
                # miloc = maloc / 2
                maloc = 1 
                miloc = 0.5
            print("maloc:", maloc)
            print("train_y_range:", self.train_y_range)
            ax.yaxis.set_major_locator( plt.MultipleLocator(maloc) )
            ax.yaxis.set_minor_locator( plt.MultipleLocator(miloc) )
            ax.grid(axis='y', which='major', color='darkorange', alpha=1)
            ax.grid(axis='y', which='minor', color='darkorange', alpha=0.5)

            plt.title( sys.argv[1] + '/' + x_train_idx )
            manger = plt.get_current_fig_manager()
            manger.window.showMaximized()
            fig = plt.gcf()
            plt.show()
            figures_path = './' + sys.argv[1] + '/figures/'
            if not os.path.exists(figures_path):
                os.makedirs( figures_path )
            fig.savefig( figures_path + x_train_idx + '.png' )

    def predict_mean(self, test_point ):
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

if __name__ == '__main__':
    # ### From .npz load datas for gp training### #
    file_path = './' + sys.argv[1]
    npz_name = 'combined_q330.npz'
    # npz_name = sys.argv[2] 
    gp_train = np.load( file_path + '/' + npz_name)

    # all dimensions: x, y, z, vx, vy, vz
    # x_idx_list = [i for i in gp_train.keys()][:6]
    # y_idx_list = [i for i in gp_train.keys()][6:]
    # for i in range(len(x_idx_list)):
    #     x_train_idx = x_idx_list[i]
    #     y_train_idx = y_idx_list[i]
    #     print("***************************")
    #     print("x_train_idx: {}".format(x_train_idx) )
    #     print("y_train_idx: {}".format(y_train_idx) )

    #     gpMPC = GpMeanCombine(x_train_idx, y_train_idx, file_path, npz_name)
    #     gpMPC.predict_test()
    #     gpMPC.plot_predict_result()

    # one dimesion
    x_train_idx = sys.argv[2]
    y_train_idx = sys.argv[3]
    gpMPC = GpMeanCombine(x_train_idx, y_train_idx, file_path, npz_name)
    gpMPC.predict_test()
    gpMPC.plot_predict_result()

    # test one point
    # file_path ="/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gaussian_process/gpr/q330"
    # npz_name ="combined_q330.npz"
    # gpMPCVx = GpMeanCombine('vx', 'y_vx', file_path, npz_name)

    # t1 = time.time()
    # a = np.array([4.68015e-310])
    # x_1 = gpMPCVx.predict_mean(a)
    # print("x1: {}".format(x_1))
    # t2 = time.time()
    # print("first training time is: ", (t2 - t1))