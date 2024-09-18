import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import yaml
import scipy
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import pandas as pd
import trajectory_planning_helpers as tph
import sys

class CentreLine:
	def __init__(self, track):
		self.track = track[::2]
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

# Constants
TRACK_WIDTH_MARGIN = 0.0 # Extra Safety margin, in meters

# Modified from https://github.com/CL2-UWaterloo/Head-to-Head-Autonomous-Racing/blob/main/gym/f110_gym/envs/laser_models.py
# load map image
def getCentreLine(map_name):
	if os.path.exists(f"maps/{map_name}.png"):
		map_img_path = f"maps/{map_name}.png"
	elif os.path.exists(f"maps/{map_name}.pgm"):
		map_img_path = f"maps/{map_name}.pgm"
	else:
		raise Exception("Map not found!")

	map_yaml_path = f"maps/{map_name}.yaml"
	raw_map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
	raw_map_img = raw_map_img.astype(np.float64)

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

	# grayscale -> binary. Converts grey to black
	map_img = raw_map_img.copy()
	map_img[map_img <= 210.] = 0
	map_img[map_img > 210.] = 1

	map_height = map_img.shape[0]
	map_width = map_img.shape[1]

	# add a black border to the map to avoid edge cases
	map_img_with_border = np.zeros((map_height + 20, map_width + 20))
	map_img_with_border[10:map_height + 10, 10:map_width + 10] = map_img

	# Calculate Euclidean Distance Transform (tells us distance to nearest wall)
	dist_transform_b = scipy.ndimage.distance_transform_edt(map_img_with_border)
	dist_transform = np.zeros((map_height, map_width))
	dist_transform = dist_transform_b[10:map_height + 10, 10:map_width + 10]

	# Threshold the distance transform to create a binary image
	# You should play around with this number. Is you say hairy lines generated, either clean the map so it is more curvy or increase this number
	THRESHOLD = 0.2 
	if map_name == "berlin":
		THRESHOLD = 0.2 # tune this value for specific maps
	elif map_name == "vegas":
		THRESHOLD = 0.4
	centers = dist_transform > THRESHOLD*dist_transform.max()
	
	centerline = skeletonize(centers)
	# # The centerline has the track width encoded

	centerline_dist = np.where(centerline, dist_transform, 0.0) #distance to closest edge
	
	startX = int((0-orig_x)/map_resolution)
	startY = int((0-orig_y)/map_resolution)
	start = (startY, startX)
	print(start)
	# Distance transform to get point closest to start on centreline
	distanceToStart_img = np.ones_like(dist_transform)
	distanceToStart_img[startY, startX] = 0
	distanceToStartTransform = scipy.ndimage.distance_transform_edt(distanceToStart_img)
	distanceToStart = np.where(centerline, distanceToStartTransform, distanceToStartTransform+200)
	start_point = np.unravel_index(np.argmin(distanceToStart, axis=None), distanceToStart.shape)
	print(start_point)
	# plt.plot(startX, startY, 'go', markersize=10)
	# plt.plot(starting_point[1], starting_point[0],'ro', markersize=10)
	# plt.imshow(distanceToStart, cmap='gray')
	# plt.show()
	print(centerline[start_point])
	starting_point = (start_point[1], start_point[0])

	sys.setrecursionlimit(20000)

	NON_EDGE = 0.0
	visited = {}
	centerline_points = []
	track_widths = []
	# DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	# If you want the other direction first
	DIRECTIONS = [(0, -1), (-1, 0),  (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1) ]

	def dfs(point):
		if (point in visited): return
		visited[point] = True
		centerline_points.append(np.array(point))
		track_widths.append(np.array([centerline_dist[point[1]][point[0]], centerline_dist[point[1]][point[0]]]))
		for direction in DIRECTIONS:
			if (centerline_dist[point[1] + direction[1]][point[0] + direction[0]] != NON_EDGE and (point[0] + direction[0], point[1] + direction[1]) not in visited):
				print(centerline[point[1] + direction[1]][point[0] + direction[0]])
				dfs((point[0] + direction[0], point[1] + direction[1]))

	dfs(starting_point)

	track_widths_np = np.array(track_widths)
	waypoints = np.array(centerline_points)
	print(f"Track widths shape: {track_widths_np.shape}, waypoints shape: {waypoints.shape}")

	# Merge track widths with waypoints
	data = np.concatenate((waypoints, track_widths_np), axis=1)

	# calculate map parameters
	orig_x = origin[0]
	orig_y = origin[1]
	# ??? Should be 0
	orig_s = np.sin(origin[2])
	orig_c = np.cos(origin[2])

	# get the distance transform
	transformed_data = data
	transformed_data *= map_resolution
	transformed_data += np.array([orig_x, orig_y, 0, 0])

	# Safety margin
	transformed_data -= np.array([0, 0, TRACK_WIDTH_MARGIN, TRACK_WIDTH_MARGIN])

	# Set the step size of the track in meters
	transformed_data = tph.interp_track.interp_track(transformed_data, 0.1)

	with open(f"maps/{map_name}_centreline.csv", 'wb') as fh:
		np.savetxt(fh, transformed_data, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
	
	# transformed_data = smooth_centre_line(transformed_data)

	# Close the loop
	# transformed_data = np.vstack([transformed_data, transformed_data[0:1]])


	# with open(f"maps/{map_name}_centreline_smooth.csv", 'wb') as fh:
	# 	np.savetxt(fh, transformed_data, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')

	raw_data = pd.read_csv(f"maps/{map_name}_centreline.csv")
	x = raw_data["# x_m"].values
	y = raw_data["y_m"].values
	wr = raw_data["w_tr_right_m"].values
	wl = raw_data["w_tr_left_m"].values

	x -= orig_x
	y -= orig_y

	x /= map_resolution
	y /= map_resolution
	plt.figure()
	plt.imshow(map_img, cmap="gray", origin="lower")
	plt.plot(x,y)
	plt.show()

def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			if not os.path.exists(f"maps/{map_name}_centreline.csv"):
				print(f"Extracting centre line for: {map_name}")
				getCentreLine(map_name)


if __name__ == "__main__":
	main()


