import numpy as np
import matplotlib.pyplot as plt
import math
import os
import yaml

for file in os.listdir('maps/'):
        if file.endswith('.png'):
            map_name = file.split('.')[0]
            map_yaml_path = f"maps/{map_name}.yaml"
            centreline = np.loadtxt(f"maps/{map_name}_wl_centreline.csv", delimiter=',')
            min_curvature1 = np.loadtxt(f"maps/{map_name}_min_curve_0.csv", delimiter=',')
            # min_curvature2 = np.loadtxt(f"maps/{map_name}_min_curve_1.csv", delimiter=',')
            # min_curvature3 = np.loadtxt(f"maps/{map_name}_min_curve_unregular_sample_0.csv", delimiter=',')
            # min_curvature4 = np.loadtxt(f"maps/{map_name}_min_curve_unregular_sample_1.csv", delimiter=',')
            map_img = plt.imread(f'maps/{map_name}.png')
            # flip the image around x axis
            map_img = np.flipud(map_img)

            # load map yaml
            with open(map_yaml_path, 'r') as yaml_stream:
                try:
                    map_metadata = yaml.safe_load(yaml_stream)
                    map_resolution = map_metadata['resolution']
                    origin = map_metadata['origin']
                except yaml.YAMLError as ex:
                    print(ex)

            min_curvature2 = min_curvature1
            min_curvature4 = min_curvature3

            orig_x = origin[0]
            orig_y = origin[1]

            x = centreline[:, 0]
            y = centreline[:, 1]
            wr = centreline[:, 2]
            wl = centreline[:, 3]
            
            minCurveX = min_curvature1[:, 0]
            minCurveY = min_curvature1[:, 1]
            minCurveX2 = min_curvature2[:, 0]
            minCurveY2 = min_curvature2[:, 1]
            minCurveX3 = min_curvature3[:, 0]
            minCurveY3 = min_curvature3[:, 1]
            minCurveX4 = min_curvature4[:, 0]
            minCurveY4 = min_curvature4[:, 1]
            
            # close the loop
            minCurveX = np.append(minCurveX, minCurveX[0])
            minCurveY = np.append(minCurveY, minCurveY[0])
            minCurveX2 = np.append(minCurveX2, minCurveX2[0])
            minCurveY2 = np.append(minCurveY2, minCurveY2[0])
            minCurveX3 = np.append(minCurveX3, minCurveX3[0])
            minCurveY3 = np.append(minCurveY3, minCurveY3[0])
            minCurveX4 = np.append(minCurveX4, minCurveX4[0])
            minCurveY4 = np.append(minCurveY4, minCurveY4[0])

            x -= orig_x
            y -= orig_y
            minCurveX -= orig_x
            minCurveY -= orig_y
            minCurveX2 -= orig_x
            minCurveY2 -= orig_y
            minCurveX3 -= orig_x
            minCurveY3 -= orig_y
            minCurveX4 -= orig_x
            minCurveY4 -= orig_y

            x /= map_resolution
            y /= map_resolution
            minCurveX /= map_resolution
            minCurveY /= map_resolution
            minCurveX2 /= map_resolution
            minCurveY2 /= map_resolution
            minCurveX3 /= map_resolution
            minCurveY3 /= map_resolution
            minCurveX4 /= map_resolution
            minCurveY4 /= map_resolution

            plt.figure(figsize=(20, 20))
            # make the figure name the map name
            plt.title(file)
            plt.imshow(map_img, cmap="gray", origin="lower")
            plt.plot(x, y, label='Centreline')
            plt.plot(minCurveX, minCurveY, label='Minimum Curvature Path')  
            # plt.plot(minCurveX2, minCurveY2, label='Minimum Curvature Path 2 Optomized')  
            # plt.plot(minCurveX3, minCurveY3, label='Minimum Curvature Path Non Regular Sample')   
            # plt.plot(minCurveX4, minCurveY4, label='Minimum Curvature Path Non Regular Sample 2 Optomized') 
            plt.legend(loc='upper right')
            # save the figure to Data folder
            plt.savefig(f"Data/{file}")
plt.show()
