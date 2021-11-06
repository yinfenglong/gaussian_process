#!/usr/bin/env python
# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt

gp_low = np.load('./q330/without_gp/appgp_luo_low.npy')
print("data_shape", gp_low.shape)
gp_high = np.load('./q330/without_gp/appgp_luo_high.npy')
print("data_shape", gp_high.shape)
gp_mix = np.load('./q330/without_gp/appgp_luo_mix.npy')
print("data_shape", gp_mix.shape)

data_length = gp_low.shape[0]
t = 0.01 * np.arange(data_length-310)
def plot_pose_traj_gp_three( gp_acc, tag, gp_offline_i, gp_offline_3 ):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(t, gp_acc,'m-.')
    plt.plot(t, gp_offline_i, 'k:')
    plt.plot(t, gp_offline_3, 'g--')
    plt.legend([
        'gp_offline_low',
        'gp_offline_high',
        'gp_offline_combine'
    ], loc=2)
    # ax2.set_ylim( [-0.2,0.2])
    data_range = np.max(gp_acc) - np.min(gp_acc)
    if data_range < 0.3:
        maloc = 0.02 
        miloc = 0.01
    elif data_range < 2:
        maloc = 0.2 
        miloc = 0.1
    elif data_range > 2:
        # maloc = float( '%.1f'%(train_y_range/30))
        # miloc = maloc / 2
        maloc = 1 
        miloc = 0.2

    # y_grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(maloc))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(miloc))
    ax.grid(axis='y', which='both')
    # x_grid
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='both')
    # title = np_name #sys.argv[2]
    # plt.title(title + ':' + tag)
    manger = plt.get_current_fig_manager()
    manger.window.showMaximized()
    fig = plt.gcf()
    plt.show()
    fig.savefig('../../thesis_figures/svg/' + 'figures' + '_' + tag + '.svg', format='svg', dpi=800 )

plot_pose_traj_gp_three( gp_low[310:,0], 'x', gp_high[310:,0], gp_mix[310:,0])
plot_pose_traj_gp_three( gp_low[310:,1], 'y', gp_high[310:,1], gp_mix[310:,1])
plot_pose_traj_gp_three( gp_low[310:,2], 'z', gp_high[310:,2], gp_mix[310:,2])