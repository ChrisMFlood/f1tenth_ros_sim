import trajectory_planning_helpers as tph
from argparse import Namespace
import numpy as np
import os
import cv2 as cv
import yaml

def load_parameter_file(paramFile):
	file_name = f"params/{paramFile}.yaml"
	with open(file_name, 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)    
	return params

class Track:
	def __init__(self, track):
		self.track = track
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

def generateVelocityProfile(track):
	'''
	generate velocity profile for the given track
	
	.. inputs::
    :param track:           track in the format [x, y, w_tr_right, w_tr_left, (banking)].
    :type track:            np.ndarray
	'''
	track = Track(track)
	racetrack_params = load_parameter_file("RaceTrackGenerator")
	vehicle_params = load_parameter_file("vehicle_params")
	ax_max_machine = np.array([[0, racetrack_params["max_longitudinal_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"]]])
	mu = racetrack_params["mu"]* np.ones(len(track.path))
	ggv = np.array([[0, racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]], [vehicle_params["max_speed"], racetrack_params["max_longitudinal_acc"], racetrack_params["max_lateral_acc"]]])
	
	speeds = tph.calc_vel_profile.calc_vel_profile(ax_max_machines=ax_max_machine, kappa=track.kappa, el_lengths=track.el_lengths, 
												closed=False, drag_coeff=0, m_veh=vehicle_params["vehicle_mass"], ggv=ggv, mu=mu, 
												v_max=vehicle_params["max_speed"], v_start=vehicle_params["max_speed"])
	acceleration = tph.calc_ax_profile.calc_ax_profile(speeds, track.el_lengths, True)
	t = tph.calc_t_profile.calc_t_profile(speeds, track.el_lengths, 0, acceleration)
	print(t[-1])
	return speeds, acceleration, t


def main():
	for file in os.listdir('maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			track = np.loadtxt(f"maps/{map_name}_centreline.csv", delimiter=',', skiprows=1)
			v,a,t=generateVelocityProfile(track)
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_0.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_1.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_2.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_3.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_4.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_iqp.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_min_curve_short_iqp.csv")
			# v,a,t=generateVelocityProfile(f"maps/{map_name}_short.csv")
			print(v,a,t)

if __name__ == "__main__":
	main()