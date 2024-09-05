import trajectory_planning_helpers as tph

from argparse import Namespace
import numpy as np
import os
import cv2 as cv
from map_converter import getCentreLine
import yaml

class CentreLine:
	def __init__(self, file_name):
		self.track = np.loadtxt(file_name, delimiter=',')
		# get every second point of the track to reduce the number of points
		# self.track = self.track[::10]
		self.track, self.sampled_index = tph.nonreg_sampling.nonreg_sampling(self.track, 0.0000000000001, 40)

		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
		



def generateMinCurvaturePath(map_name):
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	# vehicle_params = load_parameter_file("vehicle_params")
	# centreline_file = f"maps/{map_name}_min_curve_2.csv"
	centreline_file = f"maps/{map_name}_wl_centreline.csv"
	centreline = CentreLine(centreline_file)
	
	closed_path = np.row_stack([centreline.path, centreline.path[0]])
	closed_lengths = np.append(centreline.el_lengths, centreline.el_lengths[0])
	# print(closed_lengths)
	coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(closed_path, closed_lengths)
	# print(A)
	widths = centreline.widths - racetrack_params["vehicle_width"] / 2
	track = np.concatenate([centreline.path, widths], axis=1)
	# print(track)
	alpha, error = tph.opt_min_curv.opt_min_curv(track, centreline.normvectors, A, 1, 0, print_debug=True, closed=True)
	path, _, _, _, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, _, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(centreline.path, centreline.normvectors, alpha, racetrack_params["raceline_step"]) 
	# print(path)
	psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(path, el_lengths_raceline_interp_cl, True)
	min_curve_line = np.concatenate([s_raceline[:, None], path, psi[:, None], kappa[:, None]], axis=1)
	# with open(f"maps/{map_name}_min_curve_data3.csv", 'wb') as fh:
	# 	np.savetxt(fh, min_curve_line, fmt='%0.16f', delimiter=',')

	centreline.widths[:, 0] -= alpha
	centreline.widths[:, 1] += alpha
	new_widths = tph.interp_track_widths.interp_track_widths(centreline.widths, spline_inds_raceline_interp, t_values_raceline_interp)
	min_curve_track = np.concatenate([path, new_widths], axis=1)
	with open(f"maps/{map_name}_min_curve.csv", 'wb') as fh:
		np.savetxt(fh, min_curve_track, fmt='%0.16f', delimiter=',')


def load_parameter_file(planner_name):
	file_name = f"params/{planner_name}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			if not os.path.exists(f"maps/{map_name}_wl_centreline.csv"):
				print(f"Extracting centre line for: {map_name}")
				getCentreLine(map_name)
			if not os.path.exists(f"maps/{map_name}_min_curve_0.csv"):
				print(f"Generating minimum curvature path for: {map_name}")
				generateMinCurvaturePath(map_name)
	# map_name = "gbr"
	# print(f"Generating minimum curvature path for: {map_name}")
	# generateMinCurvaturePath(map_name)

if __name__ == "__main__":
	main()