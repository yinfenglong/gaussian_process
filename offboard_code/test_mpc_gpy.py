#!/usr/bin/env python
# coding=utf-8
import numpy as np
from gpr.mpc_GPyTorch_predict import GpMean 
import time

gpMPCVx = GpMean('vx','y_vx')

t1 = time.time()
a = np.array([4.68015e-310])
x_1 = gpMPCVx.predict_mean(a, gpMPCVx.likelihood_pred, gpMPCVx.model_to_predict)
t2 = time.time()
print("first training time is: ", (t2 - t1))

t3 = time.time()
b = np.array([2.68015e-310])
x_2 = gpMPCVx.predict_mean(b, gpMPCVx.likelihood_pred, gpMPCVx.model_to_predict)
t4 = time.time()
print("second training time is: ", (t4 - t3))

t5 = time.time()
c = np.array([1.68015e-310])
x_3 = gpMPCVx.predict_mean(c, gpMPCVx.likelihood_pred, gpMPCVx.model_to_predict)
t6 = time.time()
print("second training time is: ", (t6 - t5))

print("x_1", x_1)
print("x_2", x_2)
print("x_3", x_3)


'''
x_train shape torch.Size([5892])
y_train shape torch.Size([5892])
first training time is:  0.3513493537902832
second training time is:  0.0013148784637451172
second training time is:  0.0011932849884033203
x_1 [-0.01793517]
x_2 [-0.01793517]
x_3 [-0.01793517]
'''
