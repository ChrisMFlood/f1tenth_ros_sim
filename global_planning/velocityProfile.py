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

class Track:
	def __init__(self, track_path):
		self.track = np.loadtxt(track_path, delimiter=',', skiprows=1)
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

def generateVelocityProfile(track_path):
	'''generate velocity profile for the given track'''
	map_name = track_path.split('/')[-1].split('.')[0]
	print(f"Generating velocity profile for: {map_name}")
	track = Track(track_path)
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	vehicle_params = load_parameter_file("vehicle_params")
	ax_max_machine = np.array([[0, racetrack_params["max_longitudinal_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"]]])
	mu = racetrack_params["mu"]* np.ones(len(track.path))
	ggv = np.array([[0, racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]]])
	
	speeds = tph.calc_vel_profile.calc_vel_profile(ax_max_machines=ax_max_machine, kappa=track.kappa, el_lengths=track.el_lengths, 
												closed=False, drag_coeff=0, m_veh=vehicle_params["vehicle_mass"], ggv=ggv, mu=mu, 
												v_max=vehicle_params["max_speed"], v_start=0)
	acceleration = tph.calc_ax_profile.calc_ax_profile(speeds, track.el_lengths, True)
	# print(speeds)
	data = np.column_stack((speeds, acceleration))
	save_path = f"maps/{map_name}_dynamic_profile.csv"
	# with open(save_path, 'wb') as fh:
	# 	np.savetxt(fh, data, fmt='%0.16f', delimiter=',', header='vx,ax')
	
	t = tph.calc_t_profile.calc_t_profile(speeds, track.el_lengths, 0, acceleration)
	print(t[-1])
	return speeds, acceleration, t


def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			v,a,t=generateVelocityProfile(f"maps/{map_name}_centreline.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_0.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_1.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_2.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_3.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_4.csv")
			v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_iqp.csv")
			v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_short_iqp.csv")
			v,a,t=generateVelocityProfile(f"maps/{map_name}_short.csv")

if __name__ == "__main__":
	main()