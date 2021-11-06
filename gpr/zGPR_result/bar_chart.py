#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# #################################
# 2x2 
# #################################
# x = np.arange(2)
# cpu_num_list = [5.95, 1.27]
# gpu_list = [2.73, 0.33]
# bar_width = 0.3
# tick_label = ['train', 'predict']
# plt.bar(x, cpu_num_list, bar_width, color ='r', label = 'cpu')
# plt.bar(x+bar_width, gpu_list, bar_width, color ='g', label = 'cuda:gpu')
# plt.legend()
# plt.xticks(x+bar_width/2, tick_label)

# #################################
# 3x2 
# #################################
x = np.arange(2)
first = [0.024, 1.805]
second = [0.008, 0.004]
third = [0.008, 0.004] 

bar_width = 0.3
tick_label = ['approximate GP', 'exact GP']
plt.bar(x, first, bar_width, color ='r', label = 'first coordinates')
plt.bar(x+bar_width, second, bar_width, color ='g', label = 'second coordinates')
plt.bar(x+2*bar_width, third, bar_width, color ='b', label = 'third coordinates')
plt.legend()
plt.xticks(x+bar_width/2, tick_label)

plt.savefig( '../../thesis_figures/svg/' + 'train_predict.svg', format='svg', dpi=800 )
plt.show()


