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
import utils

# Constants
TRACK_WIDTH_MARGIN = 0.0 # Extra Safety margin, in meters

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.parent = None
		# self.width = centerline_dist[self.y][self.x]

	def __eq__(self, other):
		return isinstance(other, Point) and self.x == other.x and self.y == other.y

	def __hash__(self):
		return hash((self.x, self.y))  # Allow using Point as a dictionary key

# Modified from https://github.com/CL2-UWaterloo/Head-to-Head-Autonomous-Racing/blob/main/gym/f110_gym/envs/laser_models.py
# load map image
def getCentreLine(map_name):
	'''Extracts the centreline from the map image and saves it as a csv file'''
	print(f"Extracting centre line for: {map_name}")
	if os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.png"):
		map_img_path = f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.png"
	elif os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.pgm"):
		map_img_path = f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.pgm"
	else:
		raise Exception("Map not found!")

	map_yaml_path = f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.yaml"
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
	THRESHOLD = 0.4/map_resolution
	# print(f"Threshold: {THRESHOLD}")
	# if map_name == "berlin":
	# 	THRESHOLD = 0.50 # tune this value for specific maps
	# elif map_name == "vegas":
	# 	THRESHOLD = 0.50
	# centers = dist_transform > THRESHOLD*dist_transform.max()
	centers = dist_transform > THRESHOLD
	# if map_name == 'map3' or map_name == 'fourthfloor':
	# 	THRESHOLD = 0.6
	# 	centers = dist_transform > THRESHOLD*dist_transform.max()
	# print(centers)
	# plt.imshow(map_img, origin='lower', cmap='gray')
	# plt.imshow(centers, origin='lower', cmap='gray')
	# plt.show()
	
	centerline = skeletonize(centers)
	# plt.imshow(centerline, origin='lower', cmap='gray')
	# plt.show()
	# # The centerline has the track width encoded

	centerline_dist = np.where(centerline, dist_transform, 0.0) #distance to closest edge
	
	startX = int((0-orig_x)/map_resolution)
	startY = int((0-orig_y)/map_resolution)
	start = (startY, startX)
	# Distance transform to get point closest to start on centreline
	distanceToStart_img = np.ones_like(dist_transform)
	distanceToStart_img[startY, startX] = 0
	distanceToStartTransform = scipy.ndimage.distance_transform_edt(distanceToStart_img)
	distanceToStart = np.where(centerline, distanceToStartTransform, distanceToStartTransform+200)
	start_point = np.unravel_index(np.argmin(distanceToStart, axis=None), distanceToStart.shape)
	starting_point = Point(start_point[1], start_point[0])

	sys.setrecursionlimit(20000)

	NON_EDGE = 0.0
	visited = {}
	centerline_nodes = []
	track_widths = []
	DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	# If you want the other direction first
	# DIRECTIONS = [(0, -1), (-1, 0),  (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1) ]
	finalPoints = []
	

	def dfs(point: Point, parent: Point = None):
	
		if point in visited:
			return
		if not (0 <= point.x < len(centerline_dist[0]) and 0 <= point.y < len(centerline_dist)):
			return  # Avoid out-of-bounds access
	
		if centerline_dist[point.y][point.x] == NON_EDGE:
			return  # Skip non-track areas
	
		point.parent = parent
		visited[point] = True
	
		centerline_nodes.append(point)
		# track_widths.append(np.array([centerline_dist[point.y][point.x], centerline_dist[point.y][point.x]]))
	
		for direction in DIRECTIONS:
			newPoint = Point(point.x + direction[0], point.y + direction[1])
	
	
			# Check if we returned to the start
			if newPoint.x == starting_point.x and newPoint.y == starting_point.y:
				print(point.x, point.y)
				print("Returned to start")
				finalPoints.append(point)
				return
	
			if newPoint not in visited:
				dfs(newPoint, point)
				# return
			# print('done for loop')

	def iterative_dfs(starting_point: Point):
		stack = [starting_point]  # Stack stores points to visit
		visited[(starting_point.x,starting_point.y)] = True  # Track visited points and their parents
	
		while stack:
			point = stack.pop()
	
			if not (0 <= point.x < len(centerline_dist[0]) and 0 <= point.y < len(centerline_dist)):
				continue  # Avoid out-of-bounds access
			if centerline_dist[point.y][point.x] == NON_EDGE:
				continue  # Skip non-track areas
			
			centerline_nodes.append(point)
	
			for direction in DIRECTIONS:
				newPoint = Point(point.x + direction[0], point.y + direction[1])
	
	
				if not (0 <= newPoint.x < len(centerline_dist[0]) and 0 <= newPoint.y < len(centerline_dist)):
					continue  # Ensure within bounds
				if centerline_dist[newPoint.y][newPoint.x] == NON_EDGE:
					continue  # Skip non-track areas
				
				if visited.get((newPoint.x, newPoint.y)):
					if newPoint.x == starting_point.x and newPoint.y == starting_point.y and len(centerline_nodes) > 3:
						print(f"Returned to start at ({newPoint.x}, {newPoint.y})")
						newPoint.parent = point
						centerline_nodes.append(newPoint)
						# break
						return
					continue
				else:
					newPoint.parent = point
					stack.append(newPoint)
					visited[(newPoint.x, newPoint.y)] = True
					break

	iterative_dfs(starting_point)
	print(f"Number of centerline points: {len(centerline_nodes)}")

	# Ensure finalPoints is not empty before accessing
	# finalPoint = finalPoints[0]
	finalPoint = centerline_nodes[-1]
	print(f"Final point: ({finalPoint.x}, {finalPoint.y})")

	# Backtrace to reconstruct the centerline path
	centerline_points = [(finalPoint.x, finalPoint.y)]
	track_widths = [np.array([centerline_dist[finalPoint.y][finalPoint.x], centerline_dist[finalPoint.y][finalPoint.x]])]

	while finalPoint.parent is not None:
		finalPoint = finalPoint.parent
		centerline_points.append((finalPoint.x, finalPoint.y))
		track_widths.append(np.array([centerline_dist[finalPoint.y][finalPoint.x], centerline_dist[finalPoint.y][finalPoint.x]]))

	# centerline_points.reverse()  # Ensure correct start-to-end order
	# track_widths.reverse()
	
	print(f"Final Centerline Path Length: {len(centerline_points)}")

	track_widths_np = np.array(track_widths)
	waypoints = np.array(centerline_points)
	print(f"Track widths shape: {track_widths_np.shape}, waypoints shape: {waypoints.shape}")

	# Merge track widths with waypoints
	data = np.concatenate((waypoints, track_widths_np), axis=1)

	# plt.imshow(map_img, origin='lower', cmap='gray')
	# plt.plot(data[:, 0], data[:, 1], 'r.')
	# plt.show()

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

	transformed_data = utils.spline_approximation(transformed_data,4, 5, 0.05, 0.05, True)

	transformed_track = utils.trajectroy(transformed_data)

	# save trajectory
	utils.saveTrajectroy(transformed_track, map_name, 'centreline')
	
def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			# if not os.path.exists(f"maps/{map_name}_centreline.csv"):
			print(f"Extracting centre line for: {map_name}")
			getCentreLine(map_name)


if __name__ == "__main__":
	main()

