import trajectory_planning_helpers as tph
import numpy as np
import os
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import utils

def generateMinCurvaturePath(refline_path):

	map_name = refline_path.split('/')[-1].split('_')[0]
	print(f'Generating min curvature path for {map_name}')
	racetrack_params = utils.load_parameter_file("RaceTrackGenerator")

	reference_line = np.loadtxt(refline_path, delimiter=',', skiprows=1)[:,:4]
	track = utils.Trajectory_an(reference_line)
	
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
	stepsize_interp = racetrack_params["raceline_step"]
	iters_min = 5
	curv_error_allowed = 0.01

	alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp, spline_len_tmp, psi_reftrack_tmp, kappa_reftrack_tmp, dkappa_reftrack_tmp = tph.iqp_handler.iqp_handler(reftrack, normvectors, A, spline_len, psi, kappa, dkappa, kappa_bound, w_veh, print_debug, plot_debug, stepsize_interp, iters_min, curv_error_allowed)
	raceline, _,_,_, spline_inds, t_values = tph.create_raceline.create_raceline(reftrack_tmp[:, :2], normvectors_tmp, alpha_mincurv_tmp, racetrack_params["raceline_step"])[:6]
	reftrack_tmp[:, 2] -= alpha_mincurv_tmp
	reftrack_tmp[:, 3] += alpha_mincurv_tmp
	new_widths = tph.interp_track_widths.interp_track_widths(reftrack_tmp[:, 2:4], spline_inds, t_values, incl_last_point=False)
	reftrack_tmp = np.column_stack((raceline, new_widths))
	# reftrack_tmp = utils.spline_approximation(reftrack_tmp,3,5,racetrack_params["raceline_step"],racetrack_params["raceline_step"],True)
	minCurveTrack = utils.Trajectroy(reftrack_tmp, map_name)
	utils.saveTrajectroy(minCurveTrack, map_name, 'minCurve')


	





def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			generateMinCurvaturePath(f"maps/{map_name}_short.csv")




if __name__ == "__main__":
	main()