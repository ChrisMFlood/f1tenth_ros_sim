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
from velocityProfile import generateVelocityProfile
# from smoothLine import run_smoothing_process

class Track:
	def __init__(self, track):
		self.path = track[:, :2]
		self.widths = track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.path[:,1],self.path[:,0])), self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
		self.v, self.a, self.t = generateVelocityProfile(np.column_stack((self.path, self.widths)))
		self.data_save = np.column_stack((self.path, self.widths, -self.psi, self.kappa, self.s_path, self.v, self.a, self.t))

# Constants
TRACK_WIDTH_MARGIN = 0.0 # Extra Safety margin, in meters

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
	# Distance transform to get point closest to start on centreline
	distanceToStart_img = np.ones_like(dist_transform)
	distanceToStart_img[startY, startX] = 0
	distanceToStartTransform = scipy.ndimage.distance_transform_edt(distanceToStart_img)
	distanceToStart = np.where(centerline, distanceToStartTransform, distanceToStartTransform+200)
	start_point = np.unravel_index(np.argmin(distanceToStart, axis=None), distanceToStart.shape)
	starting_point = (start_point[1], start_point[0])

	sys.setrecursionlimit(20000)

	NON_EDGE = 0.0
	visited = {}
	centerline_points = []
	track_widths = []
	DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	# If you want the other direction first
	# DIRECTIONS = [(0, -1), (-1, 0),  (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1) ]

	def dfs(point):
		if (point in visited): return
		visited[point] = True
		centerline_points.append(np.array(point))
		track_widths.append(np.array([centerline_dist[point[1]][point[0]], centerline_dist[point[1]][point[0]]]))
		for direction in DIRECTIONS:
			if (centerline_dist[point[1] + direction[1]][point[0] + direction[0]] != NON_EDGE and (point[0] + direction[0], point[1] + direction[1]) not in visited):
				dfs((point[0] + direction[0], point[1] + direction[1]))

	dfs(starting_point)

	track_widths_np = np.array(track_widths)*0.9
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
	# transformed_data = tph.interp_track.interp_track(transformed_data, 0.1)
	# transformed_data = smoothLine(transformed_data)
	count=0
	while count < 3:
		transformed_data = smoothLine(transformed_data)
		count += 1
		print(f'Smoothing count: {count}')



	# Get track data
	tansformed_track = Track(transformed_data)


	# save = transformed_data
	save = tansformed_track.data_save
	with open(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_centreline.csv", 'wb') as fh:
		np.savetxt(fh, save, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time')

class CentreLine:
	def __init__(self, track, i):
		# self.track = tph.interp_track.interp_track(track, 2*i)
		self.track = tph.interp_track.interp_track(track, i)
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.closed_path = np.row_stack([self.path, self.path[0]])
		self.closed_el_lengths = np.linalg.norm(np.diff(self.closed_path, axis=0), axis=1)
		self.coeffs_x, self.coeffs_y, self.A, self.normvec_normalized = tph.calc_splines.calc_splines(self.closed_path, self.closed_el_lengths)
		self.spline_lengths = tph.calc_spline_lengths.calc_spline_lengths(self.coeffs_x, self.coeffs_y)
		self.path_interp, self.spline_inds, self.t_values, self.dists_interp = tph.interp_splines.interp_splines(self.coeffs_x, self.coeffs_y, self.spline_lengths, False, 0.1)
		self.psi, self.kappa, self.dkappa = tph.calc_head_curv_an.calc_head_curv_an(self.coeffs_x, self.coeffs_y, self.spline_inds, self.t_values, True, True)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)



def smoothLine(track):

	track = CentreLine(track, 0.1)
	reftrack = track.track
	widths = np.copy(reftrack[:, 2:4])
	reftrack[:, 2:4] = 0.1*reftrack[:, 2:4]
	normvectors = track.normvec_normalized
	A = track.A
	spline_len = track.el_lengths
	psi = track.psi
	kappa = track.kappa
	dkappa = track.dkappa
	kappa_bound = 1
	w_veh = 0
	print_debug = True
	plot_debug = False
	stepsize_interp = 0.1
	iters_min = 1
	curv_error_allowed = 100000

	alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp, spline_len_tmp, psi_reftrack_tmp, kappa_reftrack_tmp, dkappa_reftrack_tmp = tph.iqp_handler.iqp_handler(reftrack, normvectors, A, spline_len, psi, kappa, dkappa, kappa_bound, w_veh, print_debug, plot_debug, stepsize_interp, iters_min, curv_error_allowed)
	raceline, _,_,_, spline_inds, t_values = tph.create_raceline.create_raceline(reftrack_tmp[:, :2], normvectors_tmp, alpha_mincurv_tmp, 0.1)[:6]
	widths[:, 0] -= alpha_mincurv_tmp
	widths[:, 1] += alpha_mincurv_tmp
	new_widths = tph.interp_track_widths.interp_track_widths(widths, spline_inds, t_values, incl_last_point=False)
	reftrack_tmp = np.column_stack((raceline, new_widths))
	return reftrack_tmp
	


def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			# if not os.path.exists(f"maps/{map_name}_centreline.csv"):
			print(f"Extracting centre line for: {map_name}")
			getCentreLine(map_name)


if __name__ == "__main__":
	main()


