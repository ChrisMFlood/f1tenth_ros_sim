import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
import yaml
from velocityProfile import generateVelocityProfile

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
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
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
	closed_path = np.row_stack([centreline.path, centreline.path[0]])
	closed_lengths = np.linalg.norm(np.diff(closed_path, axis=0), axis=1)

	coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(closed_path, closed_lengths)
	widths = centreline.widths - racetrack_params["vehicle_width"] / 2
	track = np.concatenate([centreline.path, widths], axis=1)
	alpha = tph.opt_shortest_path.opt_shortest_path(track, centreline.normvectors, 0)
	path, _, _, _, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, _, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(centreline.path, centreline.normvectors, alpha, racetrack_params["raceline_step"])
	centreline.widths[:, 0] -= alpha
	centreline.widths[:, 1] += alpha
	new_widths = tph.interp_track_widths.interp_track_widths(centreline.widths, spline_inds_raceline_interp, t_values_raceline_interp)
	short_track = np.concatenate([path, new_widths], axis=1)
	short_track = tph.interp_track.interp_track(short_track, 0.1)
	track = Track(short_track)
	savedata = track.data_save

	save_path = f"maps/{ref}_short.csv"

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
	