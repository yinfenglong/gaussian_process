#!/usr/bin/env python
# coding=utf-8
'''
Date: 17.09.2021
Author: Yinfeng Long
usage
    python3 combine_npz.py q330
'''

import os
import sys
import numpy as np

combined = {"x": [], "y": [], "z":[], "vx":[], "vy":[], "vz":[], "y_x":[], "y_y":[], "y_z":[],\
	"y_vx":[], "y_vy":[], "y_vz":[]}

gp_path = '/home/achilles/test_ma_ws/src/itm/itm_quadrotor_node/itm_nonlinear_mpc/scripts/gpr/' + sys.argv[1]
dirs = os.listdir(gp_path)
print("dirs: {}".format(dirs))

for dir in dirs:
	files = os.listdir( (gp_path + '/' + dir) )
	for x in files:
		if os.path.splitext(x)[1] == '.npz':
			npz_path = gp_path + '/' + dir + '/' + x
			print( "npz_path: {}".format(npz_path) )
			data = np.load( npz_path )
			for key, values in data.items():
				combined[key].extend(values)
				combined_x = combined['x']
			print("shape of combined_q330_x:", np.array(combined_x).shape)
		else:
			pass

np.savez( gp_path + '/combined_' + sys.argv[1] + '.npz', \
		x=combined['x'], y=combined['y'], z=combined['z'], \
			vx = combined['vx'], vy = combined['vy'], vz = combined['vz'],\
			y_x = combined['y_x'], y_y = combined['y_y'], y_z = combined['y_z'], \
			y_vx = combined['y_vx'], y_vy = combined['y_vy'], y_vz =combined['y_vz'])
