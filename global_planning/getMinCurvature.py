import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
from map_converter import getCentreLine
import yaml

def load_parameter_file(paramFile):
	file_name = f"params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

class CentreLine:
	def __init__(self, track_path):
		track = np.loadtxt(track_path, delimiter=',', skiprows=1)
		self.track = tph.interp_track.interp_track(track, 0.2)


		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

def generateMinCurvaturePath(centreline_path, opt_number: int = 0):
	'''
	centreline_path: str, path to the centreline file (f"maps/{map_name}_wl_centreline.csv")
	'''
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	map_name = centreline_path.split('/')[-1].split('_')[0]
	centreline = CentreLine(centreline_path)
	closed_path = np.row_stack([centreline.path, centreline.path[0]])
	closed_lengths = np.append(centreline.el_lengths, centreline.el_lengths[0])

	coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(closed_path, closed_lengths)
	widths = centreline.widths - racetrack_params["vehicle_width"] / 2
	track = np.concatenate([centreline.path, widths], axis=1)
	alpha, error = tph.opt_min_curv.opt_min_curv(track, centreline.normvectors, A, 1, 0, print_debug=True, closed=True)
	path, _, _, _, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, _, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(centreline.path, centreline.normvectors, alpha, racetrack_params["raceline_step"])
	centreline.widths[:, 0] -= alpha
	centreline.widths[:, 1] += alpha
	new_widths = tph.interp_track_widths.interp_track_widths(centreline.widths, spline_inds_raceline_interp, t_values_raceline_interp)
	min_curve_track = np.concatenate([path, new_widths], axis=1)
	min_curve_track = tph.interp_track.interp_track(min_curve_track, 0.1)
	# min_curve_track = tph.nonreg_sampling.nonreg_sampling(min_curve_track, 0.1,3)

	save_path = f"maps/{map_name}_min_curve_{opt_number}.csv"

	with open(save_path, 'wb') as fh:
		np.savetxt(fh, min_curve_track, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
	
	# if opt_number == 0:
	# 	generateMinCurvaturePath(centreline_path=save_path, sample=sample, opt_number=1)



def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			# if not os.path.exists(f"maps/{map_name}_centreline.csv"):
			# 	print(f"Extracting centre line for: {map_name}")
			# 	getCentreLine(map_name)
			# if not os.path.exists(f"maps/{map_name}_min_curve_0.csv"):
			print(f"Extracting min curvature path for: {map_name}")
			generateMinCurvaturePath(centreline_path=f"maps/{map_name}_centreline.csv", opt_number=0)
			generateMinCurvaturePath(centreline_path=f"maps/{map_name}_min_curve_0.csv", opt_number=1)
			generateMinCurvaturePath(centreline_path=f"maps/{map_name}_min_curve_1.csv", opt_number=2)
			generateMinCurvaturePath(centreline_path=f"maps/{map_name}_min_curve_2.csv", opt_number=3)
			generateMinCurvaturePath(centreline_path=f"maps/{map_name}_min_curve_3.csv", opt_number=4)



if __name__ == "__main__":
	main()
	