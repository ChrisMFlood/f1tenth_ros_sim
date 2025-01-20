#!/usr/bin/env python3
import math
import numpy as np
from dataclasses import dataclass, field
import cvxpy
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.spatial import transform
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from control import utils as utils
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose

class State:
	x: float = 0.0
	y: float = 0.0
	delta: float = 0.0
	v: float = 0.0
	yaw: float = 0.0
	yawrate: float = 0.0
	beta: float = 0.0


class MPC(Node):
	""" 
	Implement Kinematic MPC on the car
	This is just a template, you are free to implement your own node!
	"""
	def __init__(self):
		super().__init__('mpc_node')
		self.get_logger().info('MPC Node Started')
		
		self.declare_parameter("odom_topic","/ego_racecar/odom")
		# self.declare_parameter("odom_topic","/pf/pose/odom")
		self.odom_topic = self.get_parameter("odom_topic").value

		self.declare_parameter("map", 'esp')
		self.declare_parameter("DTK", 0.1)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("MAX_SPEED", 8.0)
		self.declare_parameter("MIN_SPEED", -4.0)
		self.declare_parameter("MAX_ACCEL", 10)
		self.declare_parameter("MAX_STEER", 0.41)
		self.declare_parameter("MAX_DSTEER", np.pi)

		
		self.map_name = self.get_parameter("map").value
		self.DTK = self.get_parameter("DTK").value
		self.WB = self.get_parameter("wheel_base").value
		self.MAX_DSTEER = self.get_parameter("MAX_DSTEER").value
		self.MAX_SPEED = self.get_parameter("MAX_SPEED").value
		self.MIN_SPEED = self.get_parameter("MIN_SPEED").value
		self.MAX_ACCEL = self.get_parameter("MAX_ACCEL").value
		self.MAX_STEER = self.get_parameter("MAX_STEER").value



		self.declare_parameter('mass',3.74)
		self.declare_parameter('l_f',0.15875)
		self.declare_parameter('l_r',0.17145)
		self.declare_parameter('h_CoG',0.074)
		self.declare_parameter('c_f',4.718)
		self.declare_parameter('c_r',5.4562)
		self.declare_parameter('Iz',0.04712)
		self.declare_parameter('mu',1.0489)

		self.mass = self.get_parameter('mass').value
		self.l_f = self.get_parameter('l_f').value
		self.l_r = self.get_parameter('l_r').value
		self.h_CoG = self.get_parameter('h_CoG').value
		self.c_f = self.get_parameter('c_f').value
		self.c_r = self.get_parameter('c_r').value
		self.Iz = self.get_parameter('Iz').value
		self.mu = self.get_parameter('mu').value



		self.declare_parameter("NXK",7)
		self.NXK = self.get_parameter("NXK").value
		'''X = [x, y, delta, v, yaw, yaw rate, beta]'''
		self.declare_parameter("NU",2)
		self.NU = self.get_parameter("NU").value
		'''U = [acceleration, steering speed,]'''
		self.declare_parameter("TK",8)
		self.TK = self.get_parameter("TK").value
		'''Time step horizon'''

		self.Rk = np.diag([0.01, 100.0]) # input cost matrix, penalty for inputs - [accel, steering_speed]
		self.Rdk = np.diag([0.01, 100.0])  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
		self.Qk = np.diag([32.0, 32.0, 0.0, 1.0, 0.5, 0.0, 0.0])  # state error cost matrix, for the the next (T) prediction time steps 
		self.Qfk = np.diag([32.0, 32.0, 0.0, 1.0, 0.5, 0.0, 0.0])  # final state error matrix, penalty  for the final state constraints
		
		self.oa = None
		self.odelta_v = None

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
		# Publishers
		self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)
		self.ref_trajectory_pub = self.create_publisher(PoseArray, '/ref_traj', 10)
		self.predicted_path_pub = self.create_publisher(PoseArray, '/predicted_path', 10)
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_minCurve.csv', delimiter=',', skiprows=1)
		'''[x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time]'''
		utils.publishTrajectory(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints_pub)
		self.dl = np.mean(self.waypoints[1:,6] - self.waypoints[:-1,6])

		self.mpc_prob_init()


	def odom_callback(self, msg: Odometry):

		self.state = State()
		self.state.x = msg.pose.pose.position.x
		self.state.y = msg.pose.pose.position.y
		self.state.v = msg.twist.twist.linear.x
		self.state.yaw = utils.euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		self.state.yawrate = msg.twist.twist.angular.z

		self.ref_traj = self.calc_ref_trajectory(self.state, self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints[:,7])
		utils.publishTrajectory(self.ref_traj[0,:], self.ref_traj[1,:], self.ref_traj[3,:], self.ref_trajectory_pub)

		x0 = np.array([self.state.x, self.state.y, self.state.delta, self.state.v, self.state.yaw, self.state.yawrate, self.state.beta])
		self.oa,self.odelta_v,ox,oy,odelta,ov,oyaw,oyawrate,obeta,path_predict = self.linear_mpc_control_dynamic(self.ref_traj, x0, self.oa, self.odelta_v)
		# print(path_predict)
		utils.publishTrajectory(path_predict[0,:], path_predict[1,:], path_predict[3,:], self.predicted_path_pub)

		self.state.delta = self.state.delta + self.odelta_v[0]*self.DTK
		self.velocity = self.state.v + self.oa[0]*self.DTK
		# print(f'Velocity: {self.velocity}')
		utils.pubishActuation(self.state.delta, self.velocity, self.cmd_pub)


	def predict_motion_dynamic(self, x0, oa, od_v, xref):
		# path_predict = xref * 0.0
		# for i, _ in enumerate(x0):
		# 	path_predict[i, 0] = x0[i]
		path_predict = np.zeros_like(xref)
		path_predict[:, 0] = x0

		state = State()
		state.x = x0[0]
		state.y = x0[1]
		state.delta = x0[2]
		state.v = x0[3]
		state.yaw = x0[4]
		state.yawrate = x0[5]
		state.beta = x0[6]

		# for (ai, di, i) in zip(oa, od, range(1, self.TK + 1)):
		# 	state = self.update_state_kinematic(state, ai, di)
		# 	path_predict[0, i] = state.x
		# 	path_predict[1, i] = state.y
		# 	path_predict[2, i] = state.v
		# 	path_predict[3, i] = state.yaw

		for t in range(1, self.TK + 1):
			state = self.update_state_dynamic(state, oa[t-1], od_v[t-1])
			path_predict[0, t] = state.x
			path_predict[1, t] = state.y
			path_predict[2, t] = state.delta
			path_predict[3, t] = state.v
			path_predict[4, t] = state.yaw
			path_predict[5, t] = state.yawrate
			path_predict[6, t] = state.beta

		return path_predict

	def update_state_dynamic(self, state: State, a, delta_v):
		g=9.81
		state.v = max(state.v, 1e-10)
		# input check
		if delta_v >= self.MAX_DSTEER:
			delta_v = self.MAX_DSTEER
		elif delta_v <= -self.MAX_DSTEER:
			delta_v = -self.MAX_DSTEER

		if a >= self.MAX_ACCEL:
			a = self.MAX_ACCEL
		elif a <= -self.MAX_ACCEL:
			a = -self.MAX_ACCEL
		# delta = np.clip(delta, -self.MAX_STEER, self.MAX_STEER)

		K = (self.mu * self.mass) / ((self.l_f + self.l_r) * self.Iz)
		T = (g * self.l_r) - (a * self.h_CoG)
		V = (g * self.l_f) + (a * self.h_CoG)
		F = self.l_f * self.c_f
		R = self.l_r * self.c_r
		M = (self.mu * self.c_f) / (self.l_f + self.l_r)
		N = (self.mu * self.c_r) / (self.l_f + self.l_r)

		A1 = K * F * T
		A2 = K * (R * V - F * T)
		A3 = K * (self.l_f * self.l_f * self.c_f * T + self.l_r * self.l_r * self.c_r * V)
		A4 = M * T
		A5 = N * V + M * T
		A6 = N * V * self.l_r - M * T * self.l_f

		x = state.x + state.v * math.cos(state.yaw + state.beta) * self.DTK
		y = state.y + state.v * math.sin(state.yaw + state.beta) * self.DTK
		delta = state.delta + delta_v * self.DTK
		v = state.v + a * self.DTK
		yaw = (state.yaw + (state.v / self.WB) * math.tan(state.delta) * self.DTK)
		yawrate = state.yawrate+ (A1 * state.delta + A2 * state.beta - A3 * (state.yawrate / state.v)) * self.DTK
		beta = state.beta + (A4 * (state.delta / state.v) - A5 * (state.beta / state.v) + A6 * (state.yawrate / (state.v * state.v)) - state.yawrate) * self.DTK

		state.x = x
		state.y = y
		state.delta = delta
		state.v = v
		state.yaw = yaw
		state.yawrate = yawrate
		state.beta = beta

		if state.v > self.MAX_SPEED:
			state.v = self.MAX_SPEED
		elif state.v < self.MIN_SPEED:
			state.v = self.MIN_SPEED

		if delta >= self.MAX_STEER:
			delta = self.MAX_STEER
		elif delta <= -self.MAX_STEER:
			delta = -self.MAX_STEER

		return state

	def mpc_prob_solve_dynamic(self, ref_traj, path_predict, x0):
		self.x0k.value = x0

		A_block = []
		B_block = []
		C_block = []
		for t in range(self.TK):
			A, B, C = self.get_model_matrix_dynamic(
				path_predict[2, t], path_predict[3, t], path_predict[4, t], path_predict[5, t], path_predict[6, t], 0.0
			)
			A_block.append(A)
			B_block.append(B)
			C_block.extend(C)

		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(C_block)

		self.Annz_k.value = A_block.data
		self.Bnnz_k.value = B_block.data
		self.Ck_.value = C_block

		self.ref_traj_k.value = ref_traj

		# Solve the optimization problem in CVXPY
		# Solver selections: cvxpy.OSQP; cvxpy.GUROBI
		self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

		if (
			self.MPC_prob.status == cvxpy.OPTIMAL
			or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
		):
			ox = self.get_nparray_from_matrix(self.xk.value[0, :])
			oy = self.get_nparray_from_matrix(self.xk.value[1, :])
			odelta = self.get_nparray_from_matrix(self.xk.value[2, :])
			ov = self.get_nparray_from_matrix(self.xk.value[3, :])
			oyaw = self.get_nparray_from_matrix(self.xk.value[4, :])
			oyawrate = self.get_nparray_from_matrix(self.xk.value[5, :])
			obeta = self.get_nparray_from_matrix(self.xk.value[6, :])
			oa = self.get_nparray_from_matrix(self.uk.value[1, :])
			odelta_v = self.get_nparray_from_matrix(self.uk.value[0, :])

		else:
			print("Error: Cannot solve mpc..")
			oa, odelta_v, ox, oy, odelta, ov, oyaw, oyawrate, obeta = (None, None, None, None, None, None, None, None, None)

		return oa, odelta_v, ox, oy, odelta, ov, oyaw, oyawrate, obeta

	def get_nparray_from_matrix(self, x):
		return np.array(x).flatten()

	
	def linear_mpc_control_dynamic(self, ref_path, x0, oa, od_v):
		"""
		MPC contorl with updating operational point iteraitvely
		:param ref_path: reference trajectory in T steps
		:param x0: initial state vector
		:param a_old: acceleration of T steps of last time
		:param delta_old: delta of T steps of last time
		:return: acceleration and delta strategy based on current information
		"""

		if oa is None or od_v is None:
			oa = [0.0] * self.TK
			od_v = [0.0] * self.TK

		# Call the Motion Prediction function: Predict the vehicle motion for x-steps
		path_predict = self.predict_motion_dynamic(x0, oa, od_v, ref_path)
		poa, pod = oa[:], od_v[:]

		# Run the MPC optimization: Create and solve the optimization problem
		mpc_a, mpc_delta_v, mpc_x, mpc_y, mpc_delta, mpc_v, mpc_yaw, mpc_yawrate, mpc_beta = self.mpc_prob_solve_dynamic(ref_path, path_predict, x0)

		return mpc_a, mpc_delta_v, mpc_x, mpc_y, mpc_delta, mpc_v, mpc_yaw, mpc_yawrate, mpc_beta, path_predict


	def calc_ref_trajectory(self, state: State, cx, cy, cyaw, sp):
		"""
		calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
		using the current velocity, calc the T points along the reference path
		:param cx: Course X-Position
		:param cy: Course y-Position
		:param cyaw: Course Heading
		:param sp: speed profile
		:dl: distance step
		:pind: Setpoint Index
		:return: reference trajectory ref_traj, reference steering angle
		"""

		# Create placeholder Arrays for the reference trajectory for T steps
		ref_traj = np.zeros((self.NXK, self.TK + 1))
		ncourse = len(cx)

		# Find nearest index/setpoint from where the trajectories are calculated
		_, _, _, ind = utils.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

		# Load the initial parameters from the setpoint into the trajectory
		ref_traj[0, 0] = cx[ind]
		ref_traj[1, 0] = cy[ind]
		ref_traj[2, 0] = sp[ind]
		ref_traj[3, 0] = cyaw[ind]

		# based on current velocity, distance traveled on the ref line between time steps
		travel = abs(state.v) * self.DTK
		dind = travel / self.dl
		ind_list = int(ind) + np.insert(np.cumsum(np.repeat(dind, self.TK)), 0, 0).astype(int)
		ind_list[ind_list >= ncourse] -= ncourse
		ref_traj[0, :] = cx[ind_list]
		ref_traj[1, :] = cy[ind_list]
		ref_traj[2, :] = sp[ind_list]
		# cyaw[cyaw - state.yaw > 4.5] = np.abs(cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi))
		# cyaw[cyaw - state.yaw < -4.5] = np.abs(cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi))
		# print(cyaw)
		angle_thres = 4.5
		for i in ind_list:
			# self.get_logger().info(f'cyaw: {cyaw[i]}- syaw: {state.yaw} = {self.angle_diff(cyaw[i], state.yaw)}, i = {i}, {cyaw[i] - state.yaw}')
			if (cyaw[i] - state.yaw) > angle_thres:
				cyaw[i] -= 2*np.pi
				# print(cyaw[i] - state.yaw)
			if (state.yaw - cyaw[i]) > angle_thres:
				cyaw[i] += 2*np.pi
				# print(cyaw[i] - state.yaw)
		# print(cyaw)
		# print('____________________________________________________________')
		ref_traj[3, :] = cyaw[ind_list]
		return ref_traj

	def mpc_prob_init(self):
		"""
		Create MPC quadratic optimization problem using cvxpy, solver: OSQP
		Will be solved every iteration for control.
		More MPC problem information here: https://osqp.org/docs/examples/mpc.html
		More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
		"""
		# Initialize and create vectors for the optimization problem
		# Vehicle State Vector
		self.xk = cvxpy.Variable((self.NXK, self.TK + 1))
		# Control Input vector
		self.uk = cvxpy.Variable((self.NU, self.TK))

		objective = 0.0  # Objective value of the optimization problem
		constraints = []  # Create constraints array

		# Initialize reference vectors
		self.x0k = cvxpy.Parameter((self.NXK,))
		self.x0k.value = np.zeros((self.NXK,))

		# Initialize reference trajectory parameter
		self.ref_traj_k = cvxpy.Parameter((self.NXK, self.TK + 1))
		self.ref_traj_k.value = np.zeros((self.NXK, self.TK + 1))

		# Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
		R_block = block_diag(tuple([self.Rk] * self.TK))  # (2 * 8) x (2 * 8)

		# Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
		Rd_block = block_diag(tuple([self.Rdk] * (self.TK - 1)))  # (2 * 7) x (2 * 7)

		# Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
		Q_block = [self.Qk] * (self.TK)  # (4 * 8) x (4 * 8)
		Q_block.append(self.Qfk)
		Q_block = block_diag(tuple(Q_block))  # (4 * 9) x (4 * 9), Qk + Qfk

		# Formulate and create the finite-horizon optimal control problem (objective function)
		# The FTOCP has the horizon of T timesteps

		# --------------------------------------------------------
		
		# Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
		objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

		# Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
		objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

		# Objective part 3: Difference from one control input to the next control input weighted by Rd
		objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

		# --------------------------------------------------------

		# Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
		# Evaluate vehicle Dynamics for next T timesteps
		A_block = []
		B_block = []
		C_block = []
		# init path to zeros
		path_predict = np.zeros((self.NXK, self.TK + 1))
		for t in range(self.TK):  
			A, B, C = self.get_model_matrix_dynamic(path_predict[2, t], path_predict[3, t], path_predict[4, t], path_predict[5, t], path_predict[6, t], 0.0)  # reference steering angle is zero
			A_block.append(A)
			B_block.append(B)
			C_block.extend(C)

		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(C_block)
		# creating the format of matrices

		# [AA] Sparse matrix to CVX parameter for proper stuffing
		# Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
		m, n = A_block.shape  # 32, 32
		self.Annz_k = cvxpy.Parameter(A_block.nnz)  # nnz: number of nonzero elements, nnz = 128
		data = np.ones(self.Annz_k.size)  # 128 x 1, size = 128, all elements are 1
		rows = A_block.row * n + A_block.col  # No. ? element in 32 x 32 matrix
		cols = np.arange(self.Annz_k.size)  # 128 elements that need to be care - diagonal & nonzero, 4 x 4 x 8
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))	 # (rows, cols)	data

		# Setting sparse matrix data
		self.Annz_k.value = A_block.data

		# Now we use this sparse version instead of the old A_block matrix
		self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")
		# https://www.cvxpy.org/api_reference/cvxpy.atoms.affine.html#cvxpy.reshape

		# Same as A
		m, n = B_block.shape  # 32, 16 = 4 x 8, 2 x 8
		self.Bnnz_k = cvxpy.Parameter(B_block.nnz)  # nnz = 64
		data = np.ones(self.Bnnz_k.size)  # 64 = (4 x 2) x 8
		rows = B_block.row * n + B_block.col  # No. ? element in 32 x 16 matrix
		cols = np.arange(self.Bnnz_k.size)  # 0, 1, ... 63
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))  # (rows, cols)	data
		
		# sparse version instead of the old B_block
		self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
		
		# real data
		self.Bnnz_k.value = B_block.data

		# No need for sparse matrices for C as most values are parameters
		self.Ck_ = cvxpy.Parameter(C_block.shape)
		self.Ck_.value = C_block

		# -------------------------------------------------------------
		#       Add dynamics constraints to the optimization problem
		constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]
		# constraints += [cvxpy.abs(cvxpy.diff(self.uk[0, :])) <= self.MAX_DSTEER]
		constraints += [self.xk[:, 0] == self.x0k]
		# constraints += [self.xk[2, :] <= self.MAX_STEER]
		# constraints += [self.xk[2, :] >= -self.MAX_STEER]
		# constraints += [self.xk[3, :] <= self.MAX_SPEED]
		# constraints += [self.xk[3, :] >= self.MIN_SPEED]
		# constraints += [cvxpy.abs(self.uk[1, :]) <= self.MAX_ACCEL]
		# constraints += [cvxpy.abs(self.uk[1, :]) <= self.MAX_ACCEL]
		# constraints += [cvxpy.abs(self.uk[1, :]) <= self.MAX_STEER]

		# # Add constraint to ensure self.xk is valid on an occupancy grid
		# for t in range(self.TK + 1):
		# 	x_idx = cvxpy.floor(self.xk[0, t] / self.grid_resolution).astype(int)
		# 	y_idx = cvxpy.floor(self.xk[1, t] / self.grid_resolution).astype(int)
		# 	constraints += [self.occupancy_grid[x_idx, y_idx] == 0]
		# -------------------------------------------------------------

		# Create the optimization problem in CVXPY and setup the workspace
		# Optimization goal: minimize the objective function
		self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

	def get_model_matrix_dynamic(self, delta, v, yaw, yawrate, beta, a):
		"""
		Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
		Linear System: Xdot = Ax +Bu + C
		State vector: x=[x, y, delta, v, yaw, yaw rate, beta]
		:param v: speed
		:param phi: heading angle of the vehicle
		:param delta: steering angle: delta_bar
		:return: A, B, C
		"""
		v = max(v, 1e-10)
		g=9.81
		K = (self.mu * self.mass) / ((self.l_f + self.l_r) * self.Iz)
		T = (g * self.l_r) - (a * self.h_CoG)
		V = (g * self.l_f) + (a * self.h_CoG)
		F = self.l_f * self.c_f
		R = self.l_r * self.c_r
		M = (self.mu * self.c_f) / (self.l_f + self.l_r)
		N = (self.mu * self.c_r) / (self.l_f + self.l_r)

		A1 = K * F * T
		A2 = K * (R * V - F * T)
		A3 = K * (self.l_f * self.l_f * self.c_f * T + self.l_r * self.l_r * self.c_r * V)
		A4 = M * T
		A5 = N * V + M * T
		A6 = N * V * self.l_r - M * T * self.l_f

		B1 = ((-self.h_CoG * F * K) * delta+ (self.h_CoG * K * (F + R)) * beta- (self.h_CoG * K * ((self.l_r * self.l_r * self.c_r) - (self.l_f * self.l_f * self.c_f))) * (yawrate / v))
		B2 = ((-self.h_CoG * M) * (delta / v)- self.h_CoG * (N - M) * (beta / v)+ self.h_CoG * (self.l_f * M + self.l_r * N) * (yawrate / (v * v)))

		# State (or system) matrix A, 4x4
		A = np.zeros((self.NXK, self.NXK))
		A[0, 0] = 1.0
		A[1, 1] = 1.0
		A[2, 2] = 1.0
		A[3, 3] = 1.0
		A[4, 4] = 1.0
		A[5, 5] = -self.DTK * (A3 / v) + 1
		A[6, 6] = -self.DTK * A5 + 1


		# Zero row
		A[0, 3] = self.DTK * math.cos(yaw + beta)
		A[0, 4] = -self.DTK * v * math.sin(yaw + beta)
		A[0, 6] = -self.DTK * v * math.sin(yaw + beta)
		# First Row
		A[1, 3] = self.DTK * math.sin(yaw + beta)
		A[1, 4] = self.DTK * v * math.cos(yaw + beta)
		A[1, 6] = self.DTK * v * math.cos(yaw + beta)
		# Fourth Row
		A[4, 5] = self.DTK

		# Fifth Row
		A[5, 2] = self.DTK * A1
		A[5, 3] = self.DTK * A3 * (yawrate / (v * v))
		A[5, 6] = self.DTK * A2
		# Sixth Row
		A[6, 2] = self.DTK * (A4 / v)
		A[6, 3] = (self.DTK* (-A4 * beta * v + A5 * beta * v - A6 * 2 * yawrate)/ (v * v * v))
		A[6, 5] = self.DTK * ((A6 / (v * v)) - 1)

		# -------------- Input Matrix B; 7x2
		B = np.zeros((self.NXK, self.NU))
		B[2, 0] = self.DTK
		B[3, 1] = self.DTK

		B[5, 1] = self.DTK * B1
		B[6, 1] = self.DTK * B2

		# -------------- Matrix C; 7x1
		C = np.zeros(self.NXK)
		C[0] = self.DTK * (v * math.sin(yaw + beta) * yaw + v * math.sin(yaw + beta) * beta)
		C[1] = self.DTK * (-v * math.cos(yaw + beta) * yaw - v * math.cos(yaw + beta) * beta)

		C[5] = self.DTK * (-A3 * (yawrate / v) - B1 * a)
		C[6] = self.DTK * (((A4 * delta * v - A5 * beta * v + A6 * 2 * yawrate) / (v * v)) - B2 * a)

		return A, B, C








def main(args=None):
	rclpy.init(args=args)
	print("MPC Initialized")
	mpc_node = MPC()
	rclpy.spin(mpc_node)

	mpc_node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()