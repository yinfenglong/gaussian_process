#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys
import os.path
sys.path.append( os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from gpr.mpc_GPyTorch_predict import GpMean 
from gpr.gpr_GPyTorch_approximate_predict_2d import GpMeanApp2d 
from gpr.gpr_GPyTorch_predict_2d import GpMean2d 
import time

###########################
# approximate gp
###########################
# model_path_app_2d = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
# 	'/gpr/q330/20211005_1234_random_appro_2d'
# npz_name = 'data_for_gp_y.npz'
# gpMPCVx = GpMeanApp2d('vx','y_vx', 'z', model_path_app_2d, npz_name)
# gpMPCVy = GpMeanApp2d('vy','y_vy', 'z', model_path_app_2d, npz_name)
# gpMPCVz = GpMeanApp2d('vz','y_vz', 'z', model_path_app_2d, npz_name)

###########################
# exact gp
###########################
model_path_app_2d = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
	'/gpr/q330/20211005_1234_random_exact_2d'
npz_name = 'data_for_gp_y.npz'
gpMPCVx = GpMean2d('vx','y_vx', 'z', model_path_app_2d, npz_name)
gpMPCVy = GpMean2d('vy','y_vy', 'z', model_path_app_2d, npz_name)
gpMPCVz = GpMean2d('vz','y_vz', 'z', model_path_app_2d, npz_name)

height = np.array([0.5])
t1 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
b = np.c_[b, height]
a = np.c_[a, height]
c = np.c_[c, height]
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t2 = time.time()
print("first predict time is: ", (t2 - t1))

t3 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
b = np.c_[b, height]
a = np.c_[a, height]
c = np.c_[c, height]
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t4 = time.time()
print("second predict time is: ", (t4 - t3))
t3 = time.time()
a = np.array([4.68015e-310])
b = np.array([5.68015e-310])
c = np.array([7.68015e-310])
b = np.c_[b, height]
a = np.c_[a, height]
c = np.c_[c, height]
x_1 = gpMPCVx.predict_mean(a)
x_2 = gpMPCVy.predict_mean(b)
x_3 = gpMPCVz.predict_mean(c)
t4 = time.time()
print("third predict time is: ", (t4 - t3))


# model_path = os.path.join(os.path.join(os.path.dirname(__file__), '..')) + \
# 	'/gpr/q300/20210928_combine_4_random_ExactGPModel'
# npz_name = 'data_for_gp_y.npz'
# gpMPCVx = GpMean('vx','y_vx', model_path, npz_name)
# gpMPCVy = GpMean('vy','y_vy', model_path, npz_name)
# gpMPCVz = GpMean('vz','y_vz', model_path, npz_name)

# t1 = time.time()
# a = np.array([4.68015e-310])
# b = np.array([5.68015e-310])
# c = np.array([7.68015e-310])
# x_1 = gpMPCVx.predict_mean(a)
# x_2 = gpMPCVy.predict_mean(b)
# x_3 = gpMPCVz.predict_mean(c)
# t2 = time.time()
# print("first predict time is: ", (t2 - t1))

# t3 = time.time()
# a = np.array([4.68015e-310])
# b = np.array([5.68015e-310])
# c = np.array([7.68015e-310])
# x_1 = gpMPCVx.predict_mean(a)
# x_2 = gpMPCVy.predict_mean(b)
# x_3 = gpMPCVz.predict_mean(c)
# t4 = time.time()
# print("second predict time is: ", (t4 - t3))
# t3 = time.time()
# a = np.array([4.68015e-310])
# b = np.array([5.68015e-310])
# c = np.array([7.68015e-310])
# x_1 = gpMPCVx.predict_mean(a)
# x_2 = gpMPCVy.predict_mean(b)
# x_3 = gpMPCVz.predict_mean(c)
# t4 = time.time()
# print("third predict time is: ", (t4 - t3))


'''
###########################
# approximate gp
###########################
first predict time is:  0.023804664611816406
second predict time is:  0.007836103439331055
third predict time is:  0.007576942443847656

###########################
# exact gp
###########################
first predict time is:  1.8051745891571045
second predict time is:  0.003847360610961914
third predict time is:  0.0035970211029052734
'''
