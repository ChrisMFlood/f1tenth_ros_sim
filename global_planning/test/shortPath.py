import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
import yaml
from velocityProfile import generateVelocityProfile

def load_parameter_file(paramFile):
	file_name = f"/home/chris/sim_ws/src/global_planning/params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

class Centre_Line:
	def __init__(self, track):
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

class CentreLine:
	def __init__(self, track_path):
		track = np.loadtxt(track_path, delimiter=',', skiprows=1)
		self.track = tph.interp_track.interp_track(track, 0.1)
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
class Track:
	def __init__(self, track , map_name):
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
	for i,point in enumerate(path):
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



def generateShortestPath(centreline_path):
	'''
	Generates the shortest path for the given centreline path
	
	centreline_path: str, path to the centreline file (f"maps/{map_name}_centreline.csv")
	'''
	print(f'Generating shortest path for {centreline_path}')
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	map_name = centreline_path.split('/')[-1].split('_')[0]
	path_type = centreline_path.split('/')[-1].split('_')[-1].split('.')[0]
	if path_type == 'centreline':
		ref = f'{map_name}'
	else:
		temp = centreline_path.split('/')[-1].split('.')[0].split('_')[1:]
		ref = f'{map_name}_{temp[-1]}'
		print(ref)
	
	centreline = CentreLine(centreline_path)

	widths = centreline.widths - racetrack_params["vehicle_width"] / 2
	track = np.concatenate([centreline.path, widths], axis=1)
	alpha = tph.opt_shortest_path.opt_shortest_path(track, centreline.normvectors, 0)
	path, _, _, _, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, _, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(centreline.path, centreline.normvectors, alpha, racetrack_params["raceline_step"])
	centreline.widths[:, 0] -= alpha
	centreline.widths[:, 1] += alpha
	new_widths = tph.interp_track_widths.interp_track_widths(centreline.widths, spline_inds_raceline_interp, t_values_raceline_interp)
	short_track = np.concatenate([path, new_widths], axis=1)
	# short_track = run_smoothing_process(short_track)
	# short_track = iqp(short_track)
	short_track = tph.interp_track.interp_track(short_track, 0.1)
	track = Track(short_track, map_name)
	savedata = track.data_save

	save_path = f"/home/chris/sim_ws/src/global_planning/maps/{ref}_short.csv"

	with open(save_path, 'wb') as fh:
		np.savetxt(fh, savedata, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time')




def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			print(f"Extracting min curvature path for: {map_name}")
			# generateMinCurvaturePath(centreline_path=f"maps/{map_name}_centreline.csv", opt_number=0)
			generateShortestPath(centreline_path=f"maps/{map_name}_short.csv")




if __name__ == "__main__":
	main()
	