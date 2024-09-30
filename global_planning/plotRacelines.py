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
            centreline = np.loadtxt(f"maps/{map_name}_centreline.csv", delimiter=',')
            # iqpc = np.loadtxt(f"maps/{map_name}_min_curve_iqp.csv", delimiter=',')
            # short = np.loadtxt(f"maps/{map_name}_short.csv", delimiter=',')
            # iqps = np.loadtxt(f"maps/{map_name}_min_curve_short_iqp.csv", delimiter=',')

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

            # iqpcX = iqpc[:, 0]
            # iqpcY = iqpc[:, 1]
            # iqpcX -= orig_x
            # iqpcY -= orig_y
            # iqpcX /= map_resolution
            # iqpcY /= map_resolution

            # shortX = short[:, 0]
            # shortY = short[:, 1]
            # shortX -= orig_x
            # shortY -= orig_y
            # shortX /= map_resolution
            # shortY /= map_resolution

            # iqpsX = iqps[:, 0]
            # iqpsY = iqps[:, 1]
            # iqpsX -= orig_x
            # iqpsY -= orig_y
            # iqpsX /= map_resolution
            # iqpsY /= map_resolution
            

            plt.figure(figsize=(30,30))
            # make the figure name the map name
            plt.title(file)
            plt.imshow(map_img, cmap="gray", origin="lower")
            plt.plot(startX, startY, 'ro', label='Start')
            plt.plot(x, y, '--', label='Centreline')

            # plt.plot(iqpcX, iqpcY, label='IQPC')
            # plt.plot(shortX, shortY, label='Shortest Path')
            # plt.plot(iqpsX, iqpsY, label='IQPS')
            plt.legend(loc='upper right')
            # save the figure to Data folder


plt.show()
