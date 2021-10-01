#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys
import os.path
sys.path.append( os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from gpr.mpc_GPyTorch_predict import GpMean 
import time

model_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
	'/gpr/q300/20210928_combine_4_random_ExactGPModel'
npz_name = 'data_for_gp_y.npz'
gpMPCVx = GpMean('vx','y_vx', model_path, npz_name)
gpMPCVy = GpMean('vy','y_vy', model_path, npz_name)
gpMPCVz = GpMean('vz','y_vz', model_path, npz_name)

t1 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t2 = time.time()
print("first predict time is: ", (t2 - t1))

t3 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t4 = time.time()
print("second predict time is: ", (t4 - t3))
t3 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t4 = time.time()
print("third predict time is: ", (t4 - t3))
# t3 = time.time()
# b = np.array([2.68015e-310])
# x_2 = gpMPCVx.predict_mean(b)
# t4 = time.time()
# print("second predict time is: ", (t4 - t3))

# t5 = time.time()
# c = np.array([1.68015e-310])
# x_3 = gpMPCVx.predict_mean(c)
# t6 = time.time()
# print("third predict time is: ", (t6 - t5))

# print("x_1", x_1)
# print("x_2", x_2)
# print("x_3", x_3)


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
