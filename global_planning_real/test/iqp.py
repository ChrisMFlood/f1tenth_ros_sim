import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
from velocityProfile import generateVelocityProfile

def load_parameter_file(paramFile):
	file_name = f"/home/chris/sim_ws/src/global_planning/params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

class CentreLine:
	def __init__(self, track_path):
		track = np.loadtxt(track_path, delimiter=',', skiprows=1)[:, 0:4]
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

class Track:
	def __init__(self, track, map_name):
		self.path = track[:, :2]
		self.widths = track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		# self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.s_path = getS(map_name, self.path)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.path[:,1],self.path[:,0])), self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
		self.v, self.a, self.t = generateVelocityProfile(np.column_stack((self.path, self.widths)))
		self.data_save = np.column_stack((self.path, self.widths, -self.psi, self.kappa, self.s_path, self.v, self.a, self.t))

def getS(map_name, path):
	'''
	Get the S values for the given map name
	'''
	centreLine = np.loadtxt(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_centreline.csv", delimiter=',', skiprows=1)
	s=np.zeros(len(path))
	for i, point in enumerate(path):
		_, _, t, index = nearest_point(point, centreLine)
		s[i] = centreLine[index, 6] + t * np.linalg.norm(centreLine[index+1, :2] - centreLine[index, :2])
	return s
		
	

def nearest_point(point, trajectory):
	"""
	Return the nearest point along the given piecewise linear trajectory.

	Args:
		point (numpy.ndarray, (2, )): (x, y) of current pose
		trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
			NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

	Returns:
		nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
		nearest_dist (float): distance to the nearest point
		t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
		i (int): index of nearest point in the array of trajectory waypoints
	"""
	trajectory = trajectory[:, :2]
	diffs = trajectory[1:,:] - trajectory[:-1,:]
	l2s   = diffs[:,0]**2 + diffs[:,1]**2
	dots = np.empty((trajectory.shape[0]-1, ))
	for i in range(dots.shape[0]):
		dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
	t = dots / l2s
	t[t<0.0] = 0.0
	t[t>1.0] = 1.0
	projections = trajectory[:-1,:] + (t*diffs.T).T
	dists = np.empty((projections.shape[0],))
	for i in range(dists.shape[0]):
		temp = point - projections[i]
		dists[i] = np.sqrt(np.sum(temp*temp))
	min_dist_segment = np.argmin(dists)
	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def generateMinCurvaturePath(track_path):

	map_name = track_path.split('/')[-1].split('_')[0]
	path_type = track_path.split('/')[-1].split('_')[-1].split('.')[0]
	if path_type == 'centreline':
		ref = f'{map_name}'
	else:
		temp = track_path.split('/')[-1].split('.')[0].split('_')[-1]
		ref = f'{map_name}_{temp}'
	print(f'Generating min curvature path for {map_name}')
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
	stepsize_interp = 0.3
	iters_min = 5
	curv_error_allowed = 0.01

	alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp, spline_len_tmp, psi_reftrack_tmp, kappa_reftrack_tmp, dkappa_reftrack_tmp = tph.iqp_handler.iqp_handler(reftrack, normvectors, A, spline_len, psi, kappa, dkappa, kappa_bound, w_veh, print_debug, plot_debug, stepsize_interp, iters_min, curv_error_allowed)
	raceline, _,_,_, spline_inds, t_values = tph.create_raceline.create_raceline(reftrack_tmp[:, :2], normvectors_tmp, alpha_mincurv_tmp, racetrack_params["raceline_step"])[:6]
	reftrack_tmp[:, 2] -= alpha_mincurv_tmp
	reftrack_tmp[:, 3] += alpha_mincurv_tmp
	new_widths = tph.interp_track_widths.interp_track_widths(reftrack_tmp[:, 2:4], spline_inds, t_values, incl_last_point=False)
	reftrack_tmp = np.column_stack((raceline, new_widths))
	reftrack_tmp = tph.interp_track.interp_track(reftrack_tmp, 0.1)
	track_data = Track(reftrack_tmp, map_name)

	map_name = track_path.split('/')[-1].split('_')[0]
	path_type = track_path.split('/')[-1].split('_')[-1].split('.')[0]
	if path_type == 'centreline':
		ref = f'{map_name}'
	else:
		temp = track_path.split('/')[-1].split('.')[0].split('_')[-1]
		ref = f'{map_name}_{temp}'

	
	save_path = f"/home/chris/sim_ws/src/global_planning/maps/{ref}_minCurve.csv"
	with open(save_path, 'wb') as fh:
		np.savetxt(fh, track_data.data_save, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time')





def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			generateMinCurvaturePath(f"maps/{map_name}_short.csv")




if __name__ == "__main__":
	main()
	