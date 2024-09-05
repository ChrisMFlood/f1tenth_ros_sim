import numpy as np
import matplotlib.pyplot as plt
import math
import os
import yaml

for file in os.listdir('maps/'):
        if file.endswith('.png'):
            map_name = file.split('.')[0]
            map_yaml_path = f"maps/{map_name}.yaml"
            map_img = plt.imread(f'maps/{map_name}.png')
            centreline = np.loadtxt(f"maps/{map_name}_wl_centreline_smooth.csv", delimiter=',')
            min_curvature1 = np.loadtxt(f"maps/{map_name}_min_curve_0.csv", delimiter=',')
            min_curvature2 = np.loadtxt(f"maps/{map_name}_min_curve_1.csv", delimiter=',')
            min_curve_sampled = np.loadtxt(f"maps/{map_name}_min_curve_unregular_sample_0.csv", delimiter=',')
            min_curve_sampled2 = np.loadtxt(f"maps/{map_name}_min_curve_2.csv", delimiter=',')

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

            orig_x = origin[0]
            orig_y = origin[1]

            startX = int((0-orig_x)/map_resolution)
            startY = int((0-orig_y)/map_resolution)

            x = centreline[:, 0]
            y = centreline[:, 1]
            x -= orig_x
            y -= orig_y
            x /= map_resolution
            y /= map_resolution


            minCurveX = min_curvature1[:, 0]
            minCurveY = min_curvature1[:, 1]
            minCurveX -= orig_x
            minCurveY -= orig_y
            minCurveX /= map_resolution
            minCurveY /= map_resolution


            minCurveX2 = min_curvature2[:, 0]
            minCurveY2 = min_curvature2[:, 1]
            minCurveX2 -= orig_x
            minCurveY2 -= orig_y
            minCurveX2 /= map_resolution
            minCurveY2 /= map_resolution

            minCurveX3 = min_curve_sampled[:, 0]
            minCurveY3 = min_curve_sampled[:, 1]
            minCurveX3 -= orig_x
            minCurveY3 -= orig_y
            minCurveX3 /= map_resolution
            minCurveY3 /= map_resolution

            minCurveX4 = min_curve_sampled2[:, 0]
            minCurveY4 = min_curve_sampled2[:, 1]
            minCurveX4 -= orig_x
            minCurveY4 -= orig_y
            minCurveX4 /= map_resolution
            minCurveY4 /= map_resolution
            

            plt.figure(figsize=(20, 20))
            # make the figure name the map name
            plt.title(file)
            plt.imshow(map_img, cmap="gray", origin="lower")
            plt.plot(startX, startY, 'ro', label='Start')
            plt.plot(x, y, '--', label='Centreline')
            plt.plot(minCurveX, minCurveY, label='Min Curvature')
            plt.plot(minCurveX2, minCurveY2, label='Min Curvature 2nd Opt')
            plt.plot(minCurveX3, minCurveY3, label='Min Curvature Sampled')
            plt.plot(minCurveX4, minCurveY4, label='Min Curvature  3nd Opt')
            plt.legend(loc='upper right')
            # save the figure to Data folder
plt.show()
