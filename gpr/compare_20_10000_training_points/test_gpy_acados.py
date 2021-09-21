#!/usr/bin/env python
# coding=utf-8
import numpy as np
from gpr_GPyTorch_predict_acados import predict_mean
import time

t1 = time.time()
a = np.array([4.68015e-310])
x_1 = predict_mean( 'vx', 'y_vx', a)
t2 = time.time()
print("first training time is: ", (t2 - t1))

