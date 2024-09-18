import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
from map_converter import getCentreLine
import yaml
import matplotlib.pyplot as plt

def load_parameter_file(paramFile):
	file_name = f"params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

class CentreLine:
	def __init__(self, track_path):
		track = np.loadtxt(track_path, delimiter=',', skiprows=1)
		self.track = tph.interp_track.interp_track(track, 0.1)
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




		# self.psi, self.kappa, self.dkappa = tph.calc_head_curv_an.calc_head_curv_an(self.path, self.el_lengths, False)
		# self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

def generateMinCurvaturePath(track_path):
	track_name = track_path.split('/')[-1].split('_')[0]
	print(f'Generating min curvature path for {track_name}')
	racetrack_params = load_parameter_file("RaceTrackGenerator")

	track = CentreLine(track_path)
	reftrack = track.track
	normvectors = track.normvec_normalized
	A = track.A
	spline_len = track.el_lengths
	psi = track.psi
	kappa = track.kappa
	dkappa = track.dkappa
	kappa_bound = racetrack_params["max_kappa"]
	w_veh = racetrack_params["vehicle_width"]
	print_debug = True
	plot_debug = False
	stepsize_interp = 0.1
	iters_min = 5
	curv_error_allowed = 0.05

	alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp, spline_len_tmp, psi_reftrack_tmp, kappa_reftrack_tmp, dkappa_reftrack_tmp = tph.iqp_handler.iqp_handler(reftrack, normvectors, A, spline_len, psi, kappa, dkappa, kappa_bound, w_veh, print_debug, plot_debug, stepsize_interp, iters_min, curv_error_allowed)
	raceline, _,_,_, spline_inds, t_values = tph.create_raceline.create_raceline(reftrack_tmp[:, :2], normvectors_tmp, alpha_mincurv_tmp, racetrack_params["raceline_step"])[:6]
	reftrack_tmp[:, 0] -= alpha_mincurv_tmp
	reftrack_tmp[:, 1] += alpha_mincurv_tmp
	new_widths = tph.interp_track_widths.interp_track_widths(reftrack_tmp[:, :2], spline_inds, t_values, incl_last_point=False)
	reftrack_tmp = np.column_stack((raceline, new_widths))
	reftrack_tmp = tph.interp_track.interp_track(reftrack_tmp, 0.1)
	
	save_path = f"maps/{track_name}_min_curve_short_iqp_2.csv"
	with open(save_path, 'wb') as fh:
		np.savetxt(fh, reftrack_tmp, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')





def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			generateMinCurvaturePath(f"maps/{map_name}_min_curve_short_iqp.csv")




if __name__ == "__main__":
	main()
	