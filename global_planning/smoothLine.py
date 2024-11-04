import numpy as np
import trajectory_planning_helpers as tph
import csv
import matplotlib.pyplot as plt
from velocityProfile import generateVelocityProfile


class Track:
	def __init__(self, path, widths) -> None:
		self.path = path
		self.widths = widths
		self.calculate_track_vecs()

	def calculate_track_vecs(self):
		self.track = np.concatenate([self.path, self.widths], axis=1)
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
		self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

	def check_normals_crossing(self, widths=None):
		if widths is None:
			track = self.track
		else:
			track = np.concatenate([self.path, widths], axis=1)
		crossing = tph.check_normals_crossing.check_normals_crossing(track, self.nvecs)
		print(f"Crossing: {crossing}")

		return crossing 

	def smooth_centre_line(self):
		path_cl = np.row_stack([self.path, self.path[0]])
		el_lengths_cl = np.append(self.el_lengths, self.el_lengths[0])
		coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(path_cl, el_lengths_cl)

	
		self.alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, A, 1, 0, print_debug=True, closed=True)

		self.path, A_raceline, coeffs_x_raceline, coeffs_y_raceline, self.spline_inds_raceline_interp, self.t_values_raceline_interp, s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.path, self.nvecs, self.alpha, 1) 

		self.widths[:, 0] -= self.alpha
		self.widths[:, 1] += self.alpha
		self.widths = tph.interp_track_widths.interp_track_widths(self.widths, self.spline_inds_raceline_interp, self.t_values_raceline_interp)

		self.track = np.concatenate([self.path, self.widths], axis=1)
		self.calculate_track_vecs()

class CentreLine:
	def __init__(self, track):
		# track = np.loadtxt(track_path, delimiter=',', skiprows=1)[:,:4]
		self.track = tph.interp_track.interp_track(track, 1)
		self.path = self.track[:, :2]
		self.widths = self.track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.path[:,0],self.path[:,1])), self.el_lengths, False)
		self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

class Tracks:
	def __init__(self, track):
		self.path = track[:, :2]
		self.widths = track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.path[:,1],self.path[:,0])), self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
		self.v, self.a, self.t = generateVelocityProfile(np.column_stack((self.path, self.widths)))
		self.data_save = np.column_stack((self.path, self.widths, -self.psi, self.kappa, self.s_path, self.v, self.a, self.t))
		 
		  
# Define colors and path
periwinkle = '#CCCCFF'
sunset_orange = '#FF4500'
fresh_t = '#00FF00'
sweedish_green = '#009900'
naartjie = '#FFA600'
img_save_path = '/home/chris/sim_ws/src/global_planning/smooth_test/'

def plot_map_line(map_name, centre_line, run_n, new_track):
	plt.figure(6, figsize=(12, 12))
	plt.plot(centre_line.path[:, 0], centre_line.path[:, 1], '-', linewidth=2, color=periwinkle, label="Centre line")
	plt.plot(new_track.path[:, 0], new_track.path[:, 1], '-', linewidth=2, color=sunset_orange, label="Smoothed track")

	plot_line_and_boundaries(centre_line, fresh_t, False)
	plot_line_and_boundaries(new_track, sweedish_green, True)

	plt.gca().set_aspect('equal', adjustable='box')
	plt.legend(["Centre line", "Smoothed track"])
	# plt.savefig(img_save_path + f"Smoothing_{map_name}_{run_n}.svg")
	plt.show()

def plot_line_and_boundaries(new_track, color, normals=False):
	l1 = new_track.path + new_track.nvecs * new_track.widths[:, 0][:, None] # inner
	l2 = new_track.path - new_track.nvecs * new_track.widths[:, 1][:, None] # outer

	if normals:
		for i in range(len(new_track.path)):
			plt.plot([l1[i, 0], l2[i, 0]], [l1[i, 1], l2[i, 1]], linewidth=1, color=naartjie)

	plt.plot(l1[:, 0], l1[:, 1], linewidth=1, color=color)
	plt.plot(l2[:, 0], l2[:, 1], linewidth=1, color=color)

# 	# load map image
track_save_path = '/home/chris/sim_ws/src/global_planning/smooth_test/'
WIDTH_STEP_SIZE = 0.15
NUMBER_OF_WIDTH_STEPS = 30
def run_smoothing_process(track):
	"""
	This assumes that the track width is 0.9 m on each side of the centre line
	"""
	centre_line = CentreLine(track)
	# widths = np.ones_like(centre_line.path) * WIDTH_STEP_SIZE
	widths = centre_line.widths*WIDTH_STEP_SIZE
	track = Track(centre_line.path.copy(), widths)

	for i in range(NUMBER_OF_WIDTH_STEPS):
		track.widths += np.ones_like(track.path) * WIDTH_STEP_SIZE
		crossing = track.check_normals_crossing()
		if crossing:
			raise ValueError("Track is crossing before optimisation.: use smaller step size")

		# plot_map_line('map_name', centre_line, i, track)
		track.smooth_centre_line()
		centre_line.widths[:, 0] -= track.alpha
		centre_line.widths[:, 1] += track.alpha
		centre_line.widths = tph.interp_track_widths.interp_track_widths(centre_line.widths, track.spline_inds_raceline_interp, track.t_values_raceline_interp)

		test_widths = centre_line.widths + np.ones_like(track.path) * (NUMBER_OF_WIDTH_STEPS-i) * WIDTH_STEP_SIZE
		crossing = track.check_normals_crossing()
		if not crossing:
			print(f"No longer crossing: {i}")
			track.widths = centre_line.widths
			track.calculate_track_vecs()
			# plot_map_line(map_name, centre_line, "final", track)
			break

		print("")

	saveTrack = Tracks(track.track)

	# map_c_name = track_save_path + f"{map_name}_centerline.csv"
	# with open(map_c_name, 'w') as csvfile:
	# 	csvwriter = csv.writer(csvfile)
	# 	csvwriter.writerows(saveTrack.data_save)
	# save_path = track_save_path + f"{map_name}_centerline.csv"
	# with open(save_path, 'wb') as fh:
	# 	np.savetxt(fh, saveTrack.data_save, fmt='%0.16f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time')

	return saveTrack.data_save[:, :4]


def main():
	run_smoothing_process('esp')
	run_smoothing_process('gbr')
	run_smoothing_process('mco')
	run_smoothing_process('aut')
	run_smoothing_process('berlin')

if __name__ == "__main__":
	main()

















# import numpy as np
# import trajectory_planning_helpers as tph
# import sys
# import matplotlib.pyplot as plt


# def prep_track(reftrack_imp: np.ndarray,
#                kReg:int,
#                sReg:int,
#                stepsizePrep: float,
#                stepsizeReg: float,
#                debug: bool = True,
#                min_width: float = None) -> tuple:
#     """
#     Created by:
#     Alexander Heilmeier

#     Documentation:
#     This function prepares the inserted reference track for optimization.

#     .. inputs::
#     :param reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
#     :type reftrack_imp:                np.ndarray
#     :param kReg:                       order of B splines
#     :type kReg:                        int
#     :param sReg:                       smoothing factor (usually between 5 and 100).
#     :type sReg:                        int
#     :param stepsizePrep:               stepsize used for linear track interpolation before spline approximation
#     :type stepsizePrep:                float
#     :param stepsizeReg:                stepsize after smoothing
#     :type stepsizeReg:                 float
#     :param debug:                      boolean showing if debug messages should be printed
#     :type debug:                       bool
#     :param min_width:                  [m] minimum enforced track width (None to deactivate)
#     :type min_width:                   float

#     .. outputs::
#     :return reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
#     :rtype reftrack_interp:             np.ndarray
#     :return normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
#     :rtype normvec_normalized_interp:   np.ndarray
#     :return a_interp:                   LES coefficients when calculating the splines
#     :rtype a_interp:                    np.ndarray
#     :return coeffs_x_interp:            spline coefficients of the x-component
#     :rtype coeffs_x_interp:             np.ndarray
#     :return coeffs_y_interp:            spline coefficients of the y-component
#     :rtype coeffs_y_interp:             np.ndarray
#     """

#     # ------------------------------------------------------------------------------------------------------------------
#     # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------

#     # smoothing and interpolating reference track
#     reftrack_interp = tph.spline_approximation. \
#         spline_approximation(track=reftrack_imp,
#                              k_reg=kReg,
#                              s_reg=sReg,
#                              stepsize_prep=stepsizePrep,
#                              stepsize_reg=stepsizeReg,
#                              debug=debug)

#     # calculate splines
#     refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

#     coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
#         calc_splines(path=refpath_interp_cl)

#     # ------------------------------------------------------------------------------------------------------------------
#     # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------

#     normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=reftrack_interp,
#                                                                          normvec_normalized=normvec_normalized_interp,
#                                                                          horizon=10)

#     if normals_crossing:
#         bound_1_tmp = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
#         bound_2_tmp = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

#         plt.figure()

#         plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
#         for i in range(bound_1_tmp.shape[0]):
#             temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
#             plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

#         plt.grid()
#         ax = plt.gca()
#         ax.set_aspect("equal", "datalim")
#         plt.xlabel("east in m")
#         plt.ylabel("north in m")
#         plt.title("Error: at least one pair of normals is crossed!")

#         plt.show()

#         raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

#     # ------------------------------------------------------------------------------------------------------------------
#     # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
#     # ------------------------------------------------------------------------------------------------------------------

#     manipulated_track_width = False

#     if min_width is not None:
#         for i in range(reftrack_interp.shape[0]):
#             cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

#             if cur_width < min_width:
#                 manipulated_track_width = True

#                 # inflate to both sides equally
#                 reftrack_interp[i, 2] += (min_width - cur_width) / 2
#                 reftrack_interp[i, 3] += (min_width - cur_width) / 2

#     if manipulated_track_width:
#         print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
#               " order to match the requirements!", file=sys.stderr)

#     return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


# # testing --------------------------------------------------------------------------------------------------------------
# def main():
#     # load reference track
#     reftrack_imp = np.loadtxt("/home/chris/sim_ws/src/global_planning/maps/esp_centreline.csv", delimiter=",", skiprows=1)[:, :4]

#     # prepare track
#     reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = prep_track(reftrack_imp=reftrack_imp,
#                                                                                                          kReg=3,
#                                                                                                          sReg=100,
#                                                                                                          stepsizePrep=0.2,
#                                                                                                          stepsizeReg=0.1,
#                                                                                                          debug=True,
#                                                                                                          min_width=None)

#     # plot
#     plt.figure()

#     plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
#     for i in range(reftrack_interp.shape[0]):
#         bound_1 = reftrack_interp[i, :2] + normvec_normalized_interp[i] * reftrack_interp[i, 2]
#         bound_2 = reftrack_interp[i, :2] - normvec_normalized_interp[i] * reftrack_interp[i, 3]

#         temp = np.vstack((bound_1, bound_2))
#         plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

#     plt.grid()
#     ax = plt.gca()
#     ax.set_aspect("equal", "datalim")
#     plt.xlabel("east in m")
#     plt.ylabel("north in m")

#     plt.show()

# if __name__ == "__main__":
#     main()
