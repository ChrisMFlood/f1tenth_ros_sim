import numpy as np
import matplotlib.pyplot as plt
import math
import os

for file in os.listdir('maps/'):
        if file.endswith('.png'):
            map_name = file.split('.')[0]
            ben = np.loadtxt(f'maps/{map_name}_centerline.csv', delimiter=',')   
            waterloo = np.loadtxt(f"maps/{map_name}_wl_centreline.csv", delimiter=',')

            plt.plot(ben[:,0], ben[:,1], label='ben')
            plt.plot(waterloo[:,0], waterloo[:,1], label='waterloo')
            # legend in top right corner
            plt.legend(loc='upper right')
            plt.show()
