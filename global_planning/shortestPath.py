import numpy as np
import utils
import trajectory_planning_helpers as tph

def generateShortestPath(refline_path):
	'''
	Generate the shortest path for the given centreline
	'''

	map_name = refline_path.split('/')[-1].split('_')[0]
	print(f'Generating shortest path for {map_name}')

	racetrack_params = utils.load_parameter_file("RaceTrackGenerator")

	reference_line = np.loadtxt(refline_path, delimiter=',', skiprows=1)[:,:4]
	reftrack = utils.Track(reference_line)
	widths = reftrack.widths - racetrack_params["vehicle_width"]/2
	track = np.concatenate([reftrack.path, widths], axis=1)
	alpha = tph.opt_shortest_path.opt_shortest_path(track, reftrack.normvectors, 0)
	path, _, _, _, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, _, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(reftrack.path, reftrack.normvectors, alpha, racetrack_params["raceline_step"])
	reftrack.widths[:, 0] -= alpha
	reftrack.widths[:, 1] += alpha
	new_widths = tph.interp_track_widths.interp_track_widths(reftrack.widths, spline_inds_raceline_interp, t_values_raceline_interp)
	short_track = utils.spline_approximation(np.concatenate([path, new_widths], axis=1),3,5,racetrack_params["raceline_step"],racetrack_params["raceline_step"],True)
	shortTrack = utils.Trajectroy(short_track, map_name)
	utils.saveTrajectroy(shortTrack, map_name, 'short')

	
