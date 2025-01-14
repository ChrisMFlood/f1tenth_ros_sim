import numpy as np
import trajectory_planning_helpers as tph
import csv
import matplotlib.pyplot as plt
from velocityProfile import generateVelocityProfile
import yaml

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
	def __init__(self, track):
		self.path = track[:, :2]
		self.widths = track[:, 2:4]
		self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
		self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.path[:,1],self.path[:,0])), self.el_lengths, False)
		self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
		self.v, self.a, self.t = generateVelocityProfile(np.column_stack((self.path, self.widths)))
		self.data_save = np.column_stack((self.path, self.widths, -self.psi, self.kappa, self.s_path, self.v, self.a, self.t))
		

def smoothTrack(map_name):
	a=1

def main():
	a=1

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
