#!/usr/bin/env python
# coding=utf-8
import numpy as np
from mpc_GPyTorch_predict_combine import GpMeanCombine
import time

gpMean = GpMeanCombine('vx', './q330', 'combined_q330.npz')
t1 = time.time()
a = np.array([4.68015e-310])
x_1 = gpMean.predict_mean( a)
t2 = time.time()
print("x1: {}".format(x_1))
print("first training time is: ", (t2 - t1))

# t3 = time.time()
# b = np.array([2.68015e-310])
# x_2 = predict_mean( 'vx', 'y_vx', b)
# t4 = time.time()
# print("second training time is: ", (t4 - t3))

# t5 = time.time()
# c = np.array([1.68015e-310])
# x_3 = predict_mean( 'vx', 'y_vx', c)
# t6 = time.time()
# print("second training time is: ", (t6 - t5))

'''
x_train shape torch.Size([5892])
y_train shape torch.Size([5892])
first training time is:  2.0969440937042236
x_train shape torch.Size([5892])
y_train shape torch.Size([5892])
second training time is:  0.2690138816833496
x_train shape torch.Size([5892])
y_train shape torch.Size([5892])
second training time is:  0.2666921615600586
'''