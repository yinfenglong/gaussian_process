#!/usr/bin/env python3
# coding=utf-8
import numpy as np

def random_matrix_generator(x_min, x_max, y_min, y_max, z_min, z_max, size):
    np.random.seed(50)
    random_matrix = np.array([[0,0,0.4]])
    while (random_matrix.shape[0] < size+1):
        a = 0.01 * np.random.randint( low=[x_min, y_min, z_min], high=[x_max, y_max, z_max], size=(1,3) )
        x_next, y_next, z_next = a[0,0], a[0,1], a[0,2]
        last_index = random_matrix.shape[0] - 1
        x, y, z = random_matrix[last_index,0], random_matrix[last_index,1], random_matrix[last_index,2]
        
        # select random points
        if abs(z-z_next)<0.3:
            # calculate distance square and distance should be in (0.1~0.5)
            distance = abs( (x-x_next)**2 + (y-y_next)**2 + (z-z_next)**2 )
            if distance > 0.01 and distance < 0.25:
                # print("a",a)
                # print(" insert a[0]", a[0])
                random_matrix = np.append( random_matrix, a[0].reshape(1,3), axis=0 )
                # np.append( (random_matrix,a[i].reshape(1,3)), axis=0 )
                if (random_matrix.shape[0] >= size+1):
                    # random_matrix = np.append( random_matrix, np.array([[0, 0, 0.8], [0, 0, 0.4]]), axis=0 )
                    random_matrix = np.append( random_matrix, np.array([[0, 0, 0.8]]), axis=0 )
                    print("random_matrix:", random_matrix)
                    print("random_matrix has enough values")
                    return random_matrix 
